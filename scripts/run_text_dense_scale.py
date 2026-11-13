"""Full DuRetrieval exact/HNSW evaluation using an isolated Azusa Rust worker.

Pinned BGE-small-zh-v1.5: CLS pooling, query instruction, max 512 tokens,
fp16 GPU inference and float32 normalized storage. Never substitutes gold or drops queries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.request

MODEL = "BAAI/bge-small-zh-v1.5"
REVISION = "7999e1d3359715c523056ef9478215996d62a620"
INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def sha(path):
    """Hash exact bytes without loading large artifacts in memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    """Publish complete status JSON through an atomic file replacement."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def score(ids, relevant, k):
    """Compute binary qrels metrics, with requested cutoff and complete gold denominator."""
    if not relevant or len(ids) != len(set(ids)):
        raise ValueError("Empty gold or duplicate ranking")
    hits = [int(pid in relevant) for pid in ids[:k]]
    return {"hit": float(any(hits)), "recall": sum(hits)/len(relevant),
            "mrr": next((1/(i+1) for i,h in enumerate(hits) if h),0.0),
            "ndcg": sum(h/math.log2(i+2) for i,h in enumerate(hits)) /
                    sum(1/math.log2(i+2) for i in range(min(k,len(relevant))))}


def distribution(values):
    """Use nearest-rank quantiles; never label a batch average as per-query P95."""
    values = sorted(values)
    return {"mean":sum(values)/len(values), "p50":values[math.ceil(len(values)*.5)-1],
            "p95":values[math.ceil(len(values)*.95)-1]}


def download_model(root):
    """Download only required files at one fixed model commit and record their hashes."""
    root.mkdir()
    url = f"https://hf-mirror.com/api/models/{MODEL}/revision/{REVISION}"
    with urllib.request.urlopen(url, timeout=90) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise ValueError("Model revision mismatch")
    save(root/"metadata.json", metadata)
    result = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "model.safetensors", "README.md"):
        print("MODEL download", name, flush=True)
        path=root/name
        subprocess.run(["curl","-fLsS","--connect-timeout","20","--max-time","900","--retry","2",
                        "-o",str(path),f"https://hf-mirror.com/{MODEL}/resolve/{REVISION}/{name}"],check=True)
        result[name]=sha(path)
    save(root/"hashes.json", result)
    return result


def encode_all(texts, kind, model, tokenizer, output, device, batch_size):
    """Cache f32 vectors by row; publish rate and completed batches for monitoring."""
    import numpy as np
    import torch
    import torch.nn.functional as functional
    dim=model.config.hidden_size
    vectors=np.memmap(output/f"{kind}.f32",dtype="<f4",mode="w+",shape=(len(texts),dim))
    started=time.perf_counter()
    truncations=0
    for start in range(0,len(texts),batch_size):
        batch=texts[start:start+batch_size]
        if kind=="queries":
            batch=[INSTRUCTION+text for text in batch]
        # Count actual over-length samples rather than hiding truncation.
        lengths=tokenizer(batch, truncation=False, return_length=True, verbose=False)["length"]
        truncations+=sum(length>512 for length in lengths)
        inputs=tokenizer(batch,padding=True,truncation=True,max_length=512,return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden=model(**inputs).last_hidden_state[:,0]
            values=functional.normalize(hidden.float(),p=2,dim=1).cpu().numpy().astype("<f4")
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1,atol=.001):
            raise ValueError("Invalid embedding output")
        vectors[start:start+len(batch)]=values
        completed=start+len(batch)
        if start==0 or completed%512==0 or completed==len(texts):
            vectors.flush()
            elapsed=time.perf_counter()-started
            status={"phase":"encode_"+kind,"completed":completed,"total":len(texts),
                    "elapsed_s":elapsed,"rows_per_s":completed/elapsed,
                    "remaining_s":(len(texts)-completed)*elapsed/completed,"truncated_count":truncations}
            save(output/"progress.json",status)
            print(json.dumps(status),flush=True)
    vectors.flush()
    result={"rows":len(texts),"dimension":dim,"elapsed_s":time.perf_counter()-started,
            "truncated_count":truncations,"sha256":sha(output/f"{kind}.f32")}
    save(output/f"{kind}-encoding.json",result)
    return result


def run(args):
    """Run one graph, one exact pass, and three ef budgets; save raw observations incrementally."""
    import importlib.metadata
    import numpy as np
    import pyarrow.parquet as pq
    import requests
    import torch
    from transformers import AutoModel, AutoTokenizer

    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    old=json.loads((args.data/"report.json").read_text())
    tables={}
    for entry in old["inputs"]:
        path=args.data/(entry["repo"].split("/")[-1]+"-"+entry["url"].split("/")[-1])
        if sha(path)!=entry["sha256"]:
            raise ValueError("Dataset hash mismatch")
        rows=pq.read_table(path).to_pylist()
        if len(rows)!=entry["rows"]:
            raise ValueError("Dataset row count mismatch")
        tables[entry["url"].split("/")[-1].split("-")[0]]=rows
    corpus,queries=tables["corpus"],tables["queries"]
    ids=[str(row["id"]) for row in corpus]
    qids=[str(row["id"]) for row in queries]
    if len(set(ids))!=len(ids) or len(set(qids))!=len(qids):
        raise ValueError("Duplicate identity")
    positions={pid:i for i,pid in enumerate(ids)}
    qset=set(qids)
    gold={qid:set() for qid in qids}
    for row in tables["dev"]:
        qid,pid=str(row["qid"]),str(row["pid"])
        if qid not in qset or pid not in positions:
            raise ValueError("Gold absent from dataset")
        if row["score"]>0:
            gold[qid].add(pid)
    if any(not values for values in gold.values()):
        raise ValueError("Query without positive gold")
    save(output/"ids.json",{"corpus":ids,"queries":qids})
    model_dir=args.model_dir or output/"model"
    if args.model_dir:
        metadata=json.loads((model_dir/"metadata.json").read_text())
        model_hashes=json.loads((model_dir/"hashes.json").read_text())
        if metadata["sha"]!=REVISION or any(sha(model_dir/name)!=value for name,value in model_hashes.items()):
            raise ValueError("Prefetched model identity mismatch")
    else:
        model_hashes=download_model(model_dir)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refuse silently switching a full run to CPU")
    device="cuda:0"
    torch.cuda.set_per_process_memory_fraction(.25,0)
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    model=AutoModel.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float16).to(device).eval()
    encoding={}
    encoding["base"]=encode_all([row["text"] for row in corpus],"base",model,tokenizer,output,device,args.batch_size)
    encoding["queries"]=encode_all([row["text"] for row in queries],"queries",model,tokenizer,output,device,args.batch_size)
    del model
    torch.cuda.empty_cache()
    dim=encoding["base"]["dimension"]
    save(output/"worker-config.json",{"base":str((output/"base.f32").resolve()),"dim":dim,"count":len(ids),"m":16,"ef_construction":200})
    configuration={"model":MODEL,"revision":REVISION,"model_hashes":model_hashes,
        "pooling":"CLS","normalization":"L2 float32","inference_dtype":"float16",
        "query_instruction":INSTRUCTION,"max_tokens":512,"batch_size":args.batch_size,
        "metric":"Azusa cosine for exact and HNSW","m":16,"m_max":32,
        "ef_construction":200,"ef_search":[100,200,400],"k":100,"seed":42,
        "level_probability":.5,"insertion_order":"dataset row order","inputs":old["inputs"],
        "packages":{name:importlib.metadata.version(name) for name in ("numpy","torch","transformers","pyarrow")},
        "gpu":torch.cuda.get_device_name(),"worker_sha256":sha(args.worker),"runner_sha256":sha(__file__),
        "snapshot_manifest_sha256":sha(args.snapshot_manifest)}
    save(output/"configuration.json",configuration)
    save(output/"progress.json",{"phase":"building_graph"})
    worker_log=(output/"worker.log").open("w")
    worker=subprocess.Popen([str(args.worker),str(output/"worker-config.json")],stdout=worker_log,stderr=subprocess.STDOUT)
    save(output/"worker-process.json",{"pid":worker.pid})
    try:
        deadline=time.monotonic()+6*3600
        url=None
        while time.monotonic()<deadline:
            if worker.poll() is not None:
                raise RuntimeError(f"Worker exited {worker.returncode}; see worker.log")
            for line in (output/"worker.log").read_text().splitlines():
                if line.startswith("READY "):
                    url="http://"+line.split()[1]
                    break
            if url:
                break
            time.sleep(5)
        if not url:
            raise TimeoutError("Graph build exceeded six hours")
        session=requests.Session()
        session.trust_env=False
        health=session.get(url+"/health",timeout=30).json()
        if health["count"]!=len(ids) or health["dimension"]!=dim:
            raise ValueError("Worker full-graph identity mismatch")
        save(output/"health.json",health)
        qvectors=np.memmap(output/"queries.f32",dtype="<f4",mode="r",shape=(len(queries),dim))
        exact_ids=[]
        summaries={}
        for mode,ef in [("exact",100),("hnsw",100),("hnsw",200),("hnsw",400)]:
            key=f"{mode}-ef{ef}"
            totals={str(k):{name:0.0 for name in ("hit","recall","mrr","ndcg")} for k in (5,10,100)}
            native,http,ann=[],[],[]
            phase_start=time.perf_counter()
            with (output/f"{key}.jsonl").open("x",encoding="utf-8") as stream:
                for number,row in enumerate(queries):
                    vector=qvectors[number].tolist()
                    request={"vector":vector,"k":100,"ef":ef,"exact":mode=="exact"}
                    if number<10:
                        warm=session.post(url+"/search",json=request,timeout=120)
                        warm.raise_for_status()
                    tick=time.perf_counter()
                    response=session.post(url+"/search",json=request,timeout=120)
                    response.raise_for_status()
                    body=response.json()
                    http_ms=(time.perf_counter()-tick)*1000
                    indices=body["ids"]
                    if len(indices)!=100 or len(set(indices))!=100 or any(i<0 or i>=len(ids) for i in indices):
                        raise ValueError("Invalid worker ranking")
                    ranked=[ids[i] for i in indices]
                    quality={str(k):score(ranked,gold[str(row["id"])],k) for k in (5,10,100)}
                    neighbors={}
                    if mode=="exact":
                        exact_ids.append(indices)
                    else:
                        neighbors={str(k):len(set(indices[:k])&set(exact_ids[number][:k]))/k for k in (5,10,100)}
                        ann.append(neighbors)
                    item={"query_id":str(row["id"]),"ids":ranked,"indices":indices,"metrics":quality,
                          "native_ms":body["native_ms"],"http_client_ms":http_ms,"neighbor_recall":neighbors}
                    stream.write(json.dumps(item,ensure_ascii=False)+"\n")
                    stream.flush()
                    native.append(body["native_ms"]); http.append(http_ms)
                    for k,values in quality.items():
                        for name,value in values.items():
                            totals[k][name]+=value/len(queries)
                    if number%50==0 or number+1==len(queries):
                        status={"phase":key,"completed":number+1,"total":len(queries),"elapsed_s":time.perf_counter()-phase_start}
                        save(output/"progress.json",status)
                        print(json.dumps(status),flush=True)
            summaries[key]={"metrics":totals,"native_ms":distribution(native),"http_client_ms":distribution(http),
                            "neighbor_recall":{str(k):sum(item[str(k)] for item in ann)/len(ann) for k in (5,10,100)} if ann else None}
            save(output/f"{key}-summary.json",summaries[key])
        report={"status":"success","corpus_count":len(corpus),"query_count":len(queries),"configuration":configuration,
                "encoding":encoding,"build_ms":health["build_ms"],"summaries":summaries,"total_elapsed_s":time.perf_counter()-started,
                "limitations":["One graph/seed/order on a shared server; no statistical or SLA claim",
                               "Fixed BGE model differs from Azusa default Qwen embedding",
                               "HTTP benchmark worker uses Azusa HnswIndex; not production gateway/snapshot overhead",
                               "Precomputed query vectors; query latency excludes encoding and RAG generation",
                               "Cosine exact rankings are numerical reference, not relevance labels",
                               "Binary dev qrels; no parameter tuning on held-out test set"]}
        save(output/"report.json",report)
        save(output/"progress.json",{"phase":"complete","elapsed_s":time.perf_counter()-started})
        print("SUCCESS",flush=True)
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.kill(); worker.wait()
        worker_log.close()


def main():
    """Require unique output paths and preserve an explicit failure marker on errors."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker",type=Path,required=True)
    parser.add_argument("--snapshot-manifest",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--model-dir",type=Path,help="Reuse a verified pinned model prefetch")
    args=parser.parse_args()
    if args.output.exists():
        parser.error("Output must not exist")
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            save(args.output/"failure.json",{"status":"failed","error":repr(error)})
        raise


if __name__=="__main__":
    main()
