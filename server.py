"""python server.py  ->  http://localhost:8000  (serves viewer, POST /reconstruct = image -> .ply)"""
import io, os, sys, subprocess, tempfile, threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from PIL import Image
import reconstruct, refine, splatrender

LOCK = threading.Lock()  # one reconstruction at a time: overlapping requests crash Metal ("command encoder is already encoding")

class H(SimpleHTTPRequestHandler):
    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/matte":
            return self.matte()
        if path != "/reconstruct":
            return self.send_error(404)
        q = parse_qs(urlparse(self.path).query)
        zoom = float(q.get("zoom", ["1"])[0])
        body = self.rfile.read(int(self.headers["Content-Length"]))
        try:
            img = Image.open(io.BytesIO(body))
            with LOCK:
                g = raw = reconstruct.infer(img, zoom)
                if q.get("refine", ["1"])[0] == "1":
                    try:  # fall back to raw LAM output if no face is found / refinement fails
                        g = refine.refine(g, reconstruct.prep(img, zoom), fit_iters=200 if q.get("fit", ["1"])[0] == "1" else 0)
                    except Exception as e:
                        print("refine failed:", repr(e))
                if g is raw:
                    g = splatrender.unflare(raw)[0]
            ply = reconstruct.to_ply(g)
        except Exception as e:
            return self.send_error(500, str(e))
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(ply)))
        self.end_headers(); self.wfile.write(ply)

    def matte(self):  # Apple Vision subject lift (tools/matte) -> RGBA PNG
        body = self.rfile.read(int(self.headers["Content-Length"]))
        with tempfile.TemporaryDirectory() as d:
            i, o = os.path.join(d, "in"), os.path.join(d, "out.png")
            open(i, "wb").write(body)
            r = subprocess.run(["tools/matte", i, o], capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(o):
                return self.send_error(422, (r.stdout or r.stderr).strip()[:100] or "matte failed")
            png = open(o, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png)))
        self.end_headers(); self.wfile.write(png)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("loading model..."); reconstruct.load_model(); refine.depth_map(Image.new("RGB", (64, 64)))
    print("http://localhost:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), H).serve_forever()
