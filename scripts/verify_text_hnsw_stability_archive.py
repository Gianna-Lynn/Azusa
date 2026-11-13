"""Audit retrieved insertion-order experiments against frozen vectors, IDs and gold records."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from run_text_dense_scale import distribution, score, sha
from verify_text_dense_archive import close


def verify_cases(path, rows, reference, gold):
    """Validate diagnostic counts, ordering and lost gold IDs against actual query rankings."""
    observed=json.loads(path.read_text(encoding="utf-8"))
    losses=[]
    by_query={row["query_id"]:row for row in rows}
    for qid,row in by_query.items():
        delta=reference[qid]["metrics"]["10"]["recall"]-row["metrics"]["10"]["recall"]
        if delta>0: losses.append((delta,qid))
    losses.sort(key=lambda item:(-item[0],item[1]))
    assert observed["queries_with_positive_recall_at10_loss"]==len(losses)
    assert [case["query_id"] for case in observed["top_cases"]]==[qid for _,qid in losses[:15]]
    for case in observed["top_cases"]:
        qid=case["query_id"]; row=by_query[qid]
        assert case["lost_relevant_ids"]==sorted((set(reference[qid]["ids"][:10])&gold[qid])-set(row["ids"][:10]))
        assert case["exact_top10"]==reference[qid]["ids"][:10]
        assert case["hnsw_top10"]==row["ids"][:10]
        assert math.isclose(case["recall_at10_loss"],reference[qid]["metrics"]["10"]["recall"]-row["metrics"]["10"]["recall"],abs_tol=1e-12)
    return len(losses)


def main():
    """Verify byte identity, every query and aggregation, and export reviewable local audit files."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["archive","baseline","bm25"]: parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args(); root=args.archive; run=root/"run-001"
    assert sha(root/"stability-results.tar.gz")== (root/"stability-results.tar.gz.sha256").read_text().split()[0]
    verified_files=[]
    for line in (root/"result-files.sha256").read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        assert sha(root/name)==expected,name
        verified_files.append(name)
    report=json.loads((run/"report.json").read_text(encoding="utf-8")); cfg=report["configuration"]
    assert report["status"]=="success" and report["failures"]==[]
    assert not list(run.rglob("failure.json"))
    for name,field in [("run_text_hnsw_stability.py","runner_sha256"),("run_text_dense_scale.py","helpers_sha256"),
                       ("source-snapshot.json","source_manifest_sha256"),("source/target/release/text_retrieval_benchmark","worker_sha256")]:
        assert sha(root/name)==cfg[field]
    source=json.loads((root/"source-snapshot.json").read_text(encoding="utf-8"))
    previous=json.loads((args.baseline.parent/"source-snapshot.json").read_text(encoding="utf-8"))
    for name,value in source["files"].items():
        assert sha(root/"source"/name)==value,name
        if name!="src/bin/text_retrieval_benchmark.rs": assert value==previous["files"][name]
    assert sha(args.baseline/"report.json")==cfg["baseline_report_sha256"]
    assert sha(args.baseline/"exact-ef100.jsonl")==cfg["exact_rows_sha256"]
    for kind,value in cfg["vector_hashes"].items(): assert sha(args.baseline/f"{kind}.f32")==value
    identity=json.loads((args.baseline/"ids.json").read_text())
    ids,qids=identity["corpus"],identity["queries"]
    gold={row["query_id"]:set(row["relevant_ids"]) for row in map(json.loads,(args.bm25/"queries.jsonl").read_text(encoding="utf-8").splitlines())}
    reference={row["query_id"]:row for row in map(json.loads,(args.baseline/"exact-ef100.jsonl").read_text(encoding="utf-8").splitlines())}
    observations=0; case_counts={}; csv_rows=[]
    for ef in [100,200,400]:
        rows=list(map(json.loads,(args.baseline/f"hnsw-ef{ef}.jsonl").read_text(encoding="utf-8").splitlines()))
        case_counts[f"baseline-{ef}"]=verify_cases(run/f"baseline-ef-{ef}-cases.json",rows,reference,gold)
    assert [item["order_seed"] for item in report["instances"]]==[11,23,37]
    for instance in report["instances"]:
        seed=instance["order_seed"]; directory=run/f"order-{seed}"
        assert instance==json.loads((directory/"report.json").read_text())
        order=json.loads((directory/"order.json").read_text())
        assert len(order)==len(ids) and sorted(order)==list(range(len(ids)))
        assert sha(directory/"order.json")==instance["order_sha256"]
        assert sorted(instance["ef_execution_order"])==[100,200,400]
        for ef in [100,200,400]:
            rows=list(map(json.loads,(directory/f"ef-{ef}.jsonl").read_text(encoding="utf-8").splitlines()))
            assert [row["query_id"] for row in rows]==qids and len(rows)==2000
            for row in rows:
                qid=row["query_id"]; indices=row["indices"]
                assert len(indices)==len(set(indices))==100
                assert all(type(i) is int and 0<=i<len(ids) for i in indices)
                assert row["ids"]==[ids[i] for i in indices]
                for k in [5,10,100]:
                    close(score(row["ids"],gold[qid],k),row["metrics"][str(k)])
                    assert math.isclose(len(set(indices[:k])&set(reference[qid]["indices"][:k]))/k,row["neighbor_recall"][str(k)],abs_tol=1e-12)
                assert all(math.isfinite(row[name]) and row[name]>=0 for name in ["native_ms","http_client_ms"])
            calculated={"metrics":{str(k):{name:statistics.mean(row["metrics"][str(k)][name] for row in rows) for name in ["hit","recall","mrr","ndcg"]} for k in [5,10,100]},
                "neighbor_recall":{str(k):statistics.mean(row["neighbor_recall"][str(k)] for row in rows) for k in [5,10,100]},
                "native_ms":distribution([row["native_ms"] for row in rows]),"http_client_ms":distribution([row["http_client_ms"] for row in rows])}
            close(calculated,instance["summaries"][str(ef)])
            close(calculated,json.loads((directory/f"ef-{ef}-summary.json").read_text()))
            case_counts[f"order-{seed}-ef-{ef}"]=verify_cases(directory/f"ef-{ef}-cases.json",rows,reference,gold)
            csv_rows.append([seed,ef,calculated["metrics"]["10"]["recall"],calculated["metrics"]["10"]["ndcg"],calculated["neighbor_recall"]["100"],calculated["native_ms"]["p95"],calculated["http_client_ms"]["p95"],instance["build_ms"]])
            observations+=len(rows)
    for ef in [100,200,400]:
        evidence=[item["summaries"][str(ef)]["metrics"]["10"]["recall"] for item in report["instances"]]
        neighbors=[item["summaries"][str(ef)]["neighbor_recall"]["100"] for item in report["instances"]]
        close({"evidence_recall_at10":{"min":min(evidence),"mean":statistics.mean(evidence),"max":max(evidence)},
               "neighbor_recall_at100":{"min":min(neighbors),"mean":statistics.mean(neighbors),"max":max(neighbors)},
               "native_p95_ms_median_across_instances":statistics.median(item["summaries"][str(ef)]["native_ms"]["p95"] for item in report["instances"])},report["aggregate"][str(ef)])
    with (root/"metrics.csv").open("x",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream); writer.writerow(["order_seed","ef","evidence_recall_at10","ndcg_at10","neighbor_recall_at100","native_p95_ms","http_p95_ms","build_ms"]); writer.writerows(csv_rows)
    audit={"status":"passed","files_verified":len(verified_files),"source_files_verified":len(source["files"]),"query_observations_recomputed":observations,
           "case_counts":case_counts,"checks":["archive/file/source hashes","same core code and frozen vectors","complete permutations and original IDs","all quality and neighbor metrics","timing quantiles and aggregate ranges","diagnostic case counts and lost gold IDs"],
           "scope":"Saved artifact audit; no new search execution or semantic judgement"}
    (root/"local-verification.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"audit":audit,"aggregate":report["aggregate"],"elapsed_s":report["elapsed_s"]},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
