"""Smoke-test the real benchmark HTTP worker using a small normalized fixture."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    """Check ready health, exact ranking, HNSW coverage and request validation, then stop only our child."""
    import numpy as np
    import requests
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--shuffle",action="store_true",help="Verify row IDs survive shuffled insertion")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    np.array([[1,0],[0,1],[-1,0]],dtype="<f4").tofile(args.output/"base.f32")
    config={"base":str((args.output/"base.f32").resolve()),"dim":2,"count":3,"m":4,"ef_construction":20}
    if args.shuffle:
        (args.output/"order.json").write_text(json.dumps([2,0,1]))
        config["order"]=str((args.output/"order.json").resolve())
    (args.output/"config.json").write_text(json.dumps(config))
    with (args.output/"worker.log").open("w") as log:
        worker=subprocess.Popen([str(args.worker),str(args.output/"config.json")],stdout=log,stderr=subprocess.STDOUT)
        try:
            url=None
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                if worker.poll() is not None:
                    raise RuntimeError("Worker failed to start")
                for line in (args.output/"worker.log").read_text().splitlines():
                    if line.startswith("READY "):
                        url="http://"+line.split()[1]
                if url: break
                time.sleep(.1)
            assert url,"No ready endpoint"
            session=requests.Session(); session.trust_env=False
            assert session.get(url+"/health",timeout=5).json()["count"]==3
            for exact in [True,False]:
                response=session.post(url+"/search",json={"vector":[1,0],"k":3,"ef":20,"exact":exact},timeout=5)
                response.raise_for_status()
                assert response.json()["ids"]==[0,1,2],response.text
                assert response.json()["native_ms"]>=0
            response=session.post(url+"/search",json={"vector":[1],"k":3,"ef":20,"exact":False},timeout=5)
            assert response.status_code==400
            (args.output/"verification.json").write_text(json.dumps({"status":"passed","checks":["health","exact cosine order","HNSW cosine order","native timing","invalid dimension rejection"]},indent=2))
            print("PASS: HTTP worker health/exact/HNSW/timing/invalid-input smoke")
        finally:
            worker.terminate()
            worker.wait(timeout=10)


if __name__=="__main__":
    main()
