"""Rebuild three Azusa graphs with shuffled insertion; reuse frozen vectors and exact ranks.

No GPU encoding. Layer RNG stays 42: these are insertion-order replicates, not layer-seed replicates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

from run_text_dense_scale import distribution, save, score, sha


def cases(rows, reference, gold, queries, corpus, limit=15):
    """Rank observed evidence-recall losses; do not mistake missing neighbors for bad relevance."""
    losses=[]
    for row in rows:
        qid=row["query_id"]
        delta=reference[qid]["metrics"]["10"]["recall"]-row["metrics"]["10"]["recall"]
        if delta>0:
            missing=sorted((set(reference[qid]["ids"][:10]) & gold[qid])-set(row["ids"][:10]))
            losses.append({"query_id":qid,"question":queries[qid],"recall_at10_loss":delta,
                "neighbor_recall_at100":row["neighbor_recall"]["100"],
                "lost_relevant_ids":missing,"exact_top10":reference[qid]["ids"][:10],
                "hnsw_top10":row["ids"][:10],
                "lost_evidence":[{"id":pid,"characters":len(corpus[pid]),"preview":corpus[pid][:300]} for pid in missing]})
    losses.sort(key=lambda row:(-row["recall_at10_loss"],row["query_id"]))
    return {"queries_with_positive_recall_at10_loss":len(losses),"top_cases":losses[:limit],
            "note":"Observed loss against fixed cosine ranking; no automatic claim about truncation or semantic cause"}


def evaluate(args, root, seed, base_report, reference, gold, query_text, corpus, identities):
    """Run one fresh graph, retaining every query and using original IDs after permutation."""
    import numpy as np
    import requests
    ids,qids=identities["corpus"],identities["queries"]
    directory=root/f"order-{seed}"
    directory.mkdir()
    order=list(range(len(ids)))
    random.Random(seed).shuffle(order)
    save(directory/"order.json",order)
    dim=base_report["encoding"]["base"]["dimension"]
    config={"base":str((args.baseline/"base.f32").resolve()),"dim":dim,"count":len(ids),
            "m":16,"ef_construction":200,"order":str((directory/"order.json").resolve())}
    save(directory/"worker-config.json",config)
    qvectors=np.memmap(args.baseline/"queries.f32",dtype="<f4",mode="r",shape=(len(qids),dim))
    with (directory/"worker.log").open("w") as log:
        worker=subprocess.Popen([str(args.worker),str(directory/"worker-config.json")],stdout=log,stderr=subprocess.STDOUT)
        save(directory/"process.json",{"pid":worker.pid,"order_seed":seed})
        try:
            deadline=time.monotonic()+2*3600
            url=None
            while time.monotonic()<deadline:
                if worker.poll() is not None:
                    raise RuntimeError(f"Worker exited {worker.returncode}")
                for line in (directory/"worker.log").read_text().splitlines():
                    if line.startswith("READY "):
                        url="http://"+line.split()[1]
                if url: break
                time.sleep(5)
            if not url: raise TimeoutError("Graph build exceeded two hours")
            session=requests.Session(); session.trust_env=False
            response=session.get(url+"/health",timeout=30); response.raise_for_status()
            health=response.json()
            if health["count"]!=len(ids) or health["dimension"]!=dim:
                raise ValueError("Wrong full graph")
            save(directory/"health.json",health)
            ef_order=[100,200,400]
            random.Random(seed+1000).shuffle(ef_order)
            summaries={}
            for ef in ef_order:
                rows=[]
                phase_start=time.perf_counter()
                with (directory/f"ef-{ef}.jsonl").open("x",encoding="utf-8") as stream:
                    for number,qid in enumerate(qids):
                        request={"vector":qvectors[number].tolist(),"k":100,"ef":ef,"exact":False}
                        if number<10:
                            response=session.post(url+"/search",json=request,timeout=120); response.raise_for_status()
                        tick=time.perf_counter()
                        response=session.post(url+"/search",json=request,timeout=120); response.raise_for_status()
                        body=response.json(); http_ms=(time.perf_counter()-tick)*1000
                        indices=body["ids"]
                        if len(indices)!=100 or len(set(indices))!=100 or any(type(i) is not int or i<0 or i>=len(ids) for i in indices):
                            raise ValueError("Invalid ranking")
                        ranked=[ids[i] for i in indices]
                        neighbors={str(k):len(set(indices[:k])&set(reference[qid]["indices"][:k]))/k for k in [5,10,100]}
                        row={"query_id":qid,"indices":indices,"ids":ranked,"metrics":{str(k):score(ranked,gold[qid],k) for k in [5,10,100]},
                             "neighbor_recall":neighbors,"native_ms":body["native_ms"],"http_client_ms":http_ms}
                        stream.write(json.dumps(row,ensure_ascii=False)+"\n"); stream.flush(); rows.append(row)
                        if number%100==0 or number+1==len(qids):
                            status={"phase":"query","order_seed":seed,"ef":ef,"completed":number+1,"total":len(qids),"elapsed_s":time.perf_counter()-phase_start}
                            save(root/"progress.json",status); print(json.dumps(status),flush=True)
                summaries[str(ef)]={"metrics":{str(k):{name:statistics.mean(row["metrics"][str(k)][name] for row in rows)
                                      for name in ["hit","recall","mrr","ndcg"]} for k in [5,10,100]},
                    "neighbor_recall":{str(k):statistics.mean(row["neighbor_recall"][str(k)] for row in rows) for k in [5,10,100]},
                    "native_ms":distribution([row["native_ms"] for row in rows]),"http_client_ms":distribution([row["http_client_ms"] for row in rows])}
                save(directory/f"ef-{ef}-summary.json",summaries[str(ef)])
                save(directory/f"ef-{ef}-cases.json",cases(rows,reference,gold,query_text,corpus))
            result={"status":"success","order_seed":seed,"order_sha256":sha(directory/"order.json"),
                    "build_ms":health["build_ms"],"ef_execution_order":ef_order,"summaries":summaries}
            save(directory/"report.json",result)
            return result
        finally:
            worker.terminate()
            try: worker.wait(timeout=20)
            except subprocess.TimeoutExpired: worker.kill(); worker.wait()


def run(args):
    """Validate frozen inputs once; execute replicates serially and preserve failed cells explicitly."""
    import pyarrow.parquet as pq
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    baseline=json.loads((args.baseline/"report.json").read_text(encoding="utf-8"))
    if baseline["status"]!="success": raise ValueError("Incomplete baseline")
    for kind in ["base","queries"]:
        if sha(args.baseline/f"{kind}.f32")!=baseline["encoding"][kind]["sha256"]:
            raise ValueError("Frozen vector mismatch")
    identity=json.loads((args.baseline/"ids.json").read_text())
    exact_rows=[json.loads(line) for line in (args.baseline/"exact-ef100.jsonl").read_text(encoding="utf-8").splitlines()]
    if [row["query_id"] for row in exact_rows]!=identity["queries"]:
        raise ValueError("Exact query identities differ")
    reference={row["query_id"]:row for row in exact_rows}
    tables={}
    for entry in baseline["configuration"]["inputs"]:
        name=entry["repo"].split("/")[-1]+"-"+entry["url"].split("/")[-1]
        path=args.data/name
        if sha(path)!=entry["sha256"]: raise ValueError("Dataset mismatch")
        tables[entry["url"].split("/")[-1].split("-")[0]]=pq.read_table(path).to_pylist()
    corpus={str(row["id"]):row["text"] for row in tables["corpus"]}
    queries={str(row["id"]):row["text"] for row in tables["queries"]}
    if list(corpus)!=identity["corpus"] or list(queries)!=identity["queries"]:
        raise ValueError("Corpus order or query identity mismatch")
    gold={qid:set() for qid in queries}
    for row in tables["dev"]:
        if str(row["qid"]) not in gold or str(row["pid"]) not in corpus: raise ValueError("Gold identity absent")
        if row["score"]>0: gold[str(row["qid"])].add(str(row["pid"]))
    if any(not ids for ids in gold.values()): raise ValueError("Missing gold")
    config={"baseline_report_sha256":sha(args.baseline/"report.json"),"exact_rows_sha256":sha(args.baseline/"exact-ef100.jsonl"),
            "vector_hashes":{kind:sha(args.baseline/f"{kind}.f32") for kind in ["base","queries"]},
            "worker_sha256":sha(args.worker),"runner_sha256":sha(__file__),"helpers_sha256":sha(Path(__file__).with_name("run_text_dense_scale.py")),
            "source_manifest_sha256":sha(args.source_manifest),"order_seeds":[11,23,37],"layer_seed":42,
            "baseline_configuration":baseline["configuration"],"note":"Three insertion-order replicates; not three layer RNG seeds; no re-encoding"}
    save(root/"configuration.json",config)
    # Save existing badcases immediately, without waiting for the three rebuilds.
    for ef in [100,200,400]:
        rows=[json.loads(line) for line in (args.baseline/f"hnsw-ef{ef}.jsonl").read_text(encoding="utf-8").splitlines()]
        save(root/f"baseline-ef-{ef}-cases.json",cases(rows,reference,gold,queries,corpus))
    successes=[]; failures=[]
    for seed in [11,23,37]:
        save(root/"progress.json",{"phase":"building_graph","order_seed":seed})
        print("BUILD order seed",seed,flush=True)
        try: successes.append(evaluate(args,root,seed,baseline,reference,gold,queries,corpus,identity))
        except Exception as error:
            failure={"order_seed":seed,"error":repr(error)}; failures.append(failure)
            save(root/f"order-{seed}"/"failure.json",failure); print("FAILED",failure,flush=True)
    aggregate={}
    if successes:
        for ef in [100,200,400]:
            quality=[row["summaries"][str(ef)]["metrics"]["10"]["recall"] for row in successes]
            ann=[row["summaries"][str(ef)]["neighbor_recall"]["100"] for row in successes]
            aggregate[str(ef)]={"evidence_recall_at10":{"min":min(quality),"mean":statistics.mean(quality),"max":max(quality)},
                "neighbor_recall_at100":{"min":min(ann),"mean":statistics.mean(ann),"max":max(ann)},
                "native_p95_ms_median_across_instances":statistics.median(row["summaries"][str(ef)]["native_ms"]["p95"] for row in successes)}
    report={"status":"success" if not failures else "failed","instances":successes,"failures":failures,"aggregate":aggregate,
            "elapsed_s":time.perf_counter()-started,"configuration":config,
            "limits":["No layer RNG seed variation","Shared server; fixed sequential query order; ef execution order shuffled per graph",
                      "Instance P95 median is not pooled-query P95","Precomputed vectors and benchmark HTTP adapter; no end-to-end generation"]}
    save(root/"report.json",report); save(root/"progress.json",{"phase":"complete" if not failures else "complete_with_failures"})
    if failures: raise RuntimeError("One or more insertion-order instances failed")
    print("SUCCESS all three insertion-order instances",flush=True)


def main():
    """Require unique run output paths; leave any failed run inspectable."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["baseline","data","worker","output","source-manifest"]: parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error("Output must not exist")
    try: run(args)
    except Exception as error:
        if args.output.exists(): save(args.output/"failure.json",{"status":"failed","error":repr(error)})
        raise


if __name__=="__main__": main()
