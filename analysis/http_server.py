import http.server
import socketserver
import os

PORT = 8000

# Change the directory to the one containing the images
os.chdir("_data/precomputed_image_pairs")

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()

# TODO try to remove server
