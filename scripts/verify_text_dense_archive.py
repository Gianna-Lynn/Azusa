"""Verify a retrieved dense experiment against original bytes and BM25 gold records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tarfile

from run_text_dense_scale import distribution, score, sha


def close(actual, expected):
    """Compare nested metric structures with floating accumulation tolerance."""
    if isinstance(expected, dict):
        assert set(actual)==set(expected)
        for key in expected:
            close(actual[key], expected[key])
    elif expected is None:
        assert actual is None
    else:
        assert math.isclose(actual, expected, abs_tol=1e-9), (actual, expected)


def main():
    """Check every input identity, recompute all observations, then write a local audit and CSV."""
    import numpy as np
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--bm25", type=Path, required=True)
    args=parser.parse_args()
    root=args.archive
    expected=(root/"dense-results.tar.gz.sha256").read_text().split()[0]
    assert sha(root/"dense-results.tar.gz")==expected, "Archive checksum mismatch"
    files=[]
    for line in (root/"result-files.sha256").read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        assert sha(root/name)==expected,name
        files.append(name)
    run=root/"run-001"
    report=json.loads((run/"report.json").read_text(encoding="utf-8"))
    assert report["status"]=="success" and not (run/"failure.json").exists()
    cfg=report["configuration"]
    assert sha(root/"run_text_dense_scale.py")==cfg["runner_sha256"]
    assert sha(root/"source/target/release/text_retrieval_benchmark")==cfg["worker_sha256"]
    assert sha(root/"source-snapshot.json")==cfg["snapshot_manifest_sha256"]
    for name,expected in cfg["model_hashes"].items():
        assert sha(root/"model-prefetch"/name)==expected,name
    snapshot=json.loads((root/"source-snapshot.json").read_text(encoding="utf-8"))
    with tarfile.open(root/"azusa-source.tar.gz") as source:
        for name,expected in snapshot["files"].items():
            assert hashlib.sha256(source.extractfile(name).read()).hexdigest()==expected,name
    old=json.loads((args.bm25/"report.json").read_text())
    assert cfg["inputs"]==old["inputs"], "Different dataset than BM25 baseline"
    gold_rows=[json.loads(line) for line in (args.bm25/"queries.jsonl").read_text(encoding="utf-8").splitlines()]
    gold={row["query_id"]:set(row["relevant_ids"]) for row in gold_rows}
    identities=json.loads((run/"ids.json").read_text())
    ids,qids=identities["corpus"],identities["queries"]
    assert len(ids)==len(set(ids))==report["corpus_count"]==100001
    assert len(qids)==len(set(qids))==report["query_count"]==2000
    assert set(qids)==set(gold)
    for kind in ["base","queries"]:
        info=report["encoding"][kind]
        path=run/f"{kind}.f32"
        assert sha(path)==info["sha256"]
        assert path.stat().st_size==info["rows"]*info["dimension"]*4
        vectors=np.memmap(path,dtype="<f4",mode="r",shape=(info["rows"],info["dimension"]))
        for start in range(0,len(vectors),4096):
            batch=vectors[start:start+4096]
            assert np.isfinite(batch).all()
            assert np.allclose(np.linalg.norm(batch,axis=1),1,atol=.001)
    exact={}
    verified={}
    for key,summary in report["summaries"].items():
        rows=[json.loads(line) for line in (run/f"{key}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [row["query_id"] for row in rows]==qids
        totals={str(k):{name:0.0 for name in ("hit","recall","mrr","ndcg")} for k in (5,10,100)}
        neighbors=[]
        for row in rows:
            qid=row["query_id"]
            indices=row["indices"]
            assert len(indices)==len(set(indices))==100
            assert all(type(i) is int and 0<=i<len(ids) for i in indices)
            assert row["ids"]==[ids[i] for i in indices]
            for k in [5,10,100]:
                values=score(row["ids"],gold[qid],k)
                close(values,row["metrics"][str(k)])
                for name,value in values.items():
                    totals[str(k)][name]+=value/len(rows)
            if key.startswith("exact"):
                exact[qid]=indices
                assert row["neighbor_recall"]=={}
            else:
                values={str(k):len(set(indices[:k])&set(exact[qid][:k]))/k for k in [5,10,100]}
                close(values,row["neighbor_recall"])
                neighbors.append(values)
            assert all(math.isfinite(row[name]) and row[name]>=0 for name in ["native_ms","http_client_ms"])
        calculated={"metrics":totals,"native_ms":distribution([row["native_ms"] for row in rows]),
                    "http_client_ms":distribution([row["http_client_ms"] for row in rows]),
                    "neighbor_recall":{str(k):sum(row[str(k)] for row in neighbors)/len(neighbors) for k in [5,10,100]} if neighbors else None}
        close(calculated,summary)
        close(calculated,json.loads((run/f"{key}-summary.json").read_text()))
        verified[key]=len(rows)
    with (root/"metrics.csv").open("x",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["mode","k","hit","recall","mrr","ndcg","neighbor_recall","native_p95_ms","http_p95_ms"])
        for key,summary in report["summaries"].items():
            for k,values in summary["metrics"].items():
                writer.writerow([key,k]+[values[name] for name in ["hit","recall","mrr","ndcg"]]+[
                    summary["neighbor_recall"][k] if summary["neighbor_recall"] else "",
                    summary["native_ms"]["p95"],summary["http_client_ms"]["p95"]])
    result={"status":"passed","verified_files":files,"source_files_verified":len(snapshot["files"]),
            "observations_recomputed":verified,"checks":["archive and file SHA256","model/binary/runner/source identity",
                "same dataset as BM25","vector bytes and normalization","query and ranking identities",
                "per-query and aggregate relevance/neighbor metrics","mean/P50/P95 recomputed"],
            "scope":"Checks saved artifacts and calculations; does not re-execute model or Rust search"}
    (root/"local-verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"files":len(files),"source_files":len(snapshot["files"]),"queries":verified,"status":"passed"}))


if __name__=="__main__":
    main()
