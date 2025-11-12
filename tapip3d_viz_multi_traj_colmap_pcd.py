import os
import numpy as np
import json
import struct
import zlib
import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import http.server
import socketserver
import socket
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingTCPServer
import base64
from plyfile import PlyData
import sys

# --- (기존과 동일한 부분) ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from scene.colmap_loader import read_extrinsics_text, read_extrinsics_binary, read_points3D_binary, read_points3D_text
from utils.colmap_utils import getNerfppNorm, storePly

viz_html_path = Path(__file__).parent / "viz_multi_traj_colmap_pcd.html" # HTML 파일 이름 변경
DEFAULT_PORT = 8000

class NoFaviconHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()

def compress_and_write(filename, header, blob):
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))
    with open(filename, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(blob)

def load_ply(ply_path):
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T
    colors = colors.astype(np.float32) / 255.0
    return points, colors

# --- (기존과 거의 동일, 반환값 구조만 약간 변경) ---
def _process_single_file(npz_file):
    """단일 NPZ 파일에서 월드 좌표계 데이터와 extrinsics를 로드합니다."""
    data = np.load(npz_file)
    extrinsics = data["extrinsics_colmap"]
    trajs = data["colmap-aligned_coords"][:, :, :3]
    visib = data["visibs"].astype(bool)

    meta = {
        "totalFrames": trajs.shape[0],
        "numTrajectoryPoints": trajs.shape[1],
    }
    return trajs, visib, extrinsics, meta

# ====================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 여기가 핵심 수정 부분입니다 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ====================================================================
def process_data_with_pcd(npz_files, colmap_0_dir, output_file, static_html_file=None, fps=4):
    """ N개의 궤적 데이터와 포인트 클라우드를 첫 번째 카메라의 좌표계로 통일하여 처리합니다. """
    
    # --- 1. 기준 좌표계 설정 ---
    # 첫 번째 npz 파일의 첫 프레임 extrinsics를 기준으로 모든 것을 변환합니다.
    _, _, extrinsics_ref, _ = _process_single_file(npz_files[0])
    world_to_reference_matrix = extrinsics_ref[0]

    all_trajectories = {}
    all_metas = []

    # --- 2. N개의 궤적 데이터 처리 ---
    for i, npz_file in enumerate(npz_files):
        trajs, visib, _, meta = _process_single_file(npz_file)
        
        # 궤적을 기준 좌표계로 변환
        T = trajs.shape[0]
        normalized_trajs = np.zeros_like(trajs)
        for t in range(T):
            homogeneous_trajs = np.concatenate([trajs[t], np.ones((trajs.shape[1], 1))], axis=1)
            transformed_trajs = (world_to_reference_matrix @ homogeneous_trajs.T).T
            normalized_trajs[t] = transformed_trajs[:, :3]
            
        # 시각화 라이브러리 좌표계에 맞춤 (Y, Z 축 반전)
        normalized_trajs[:, :, 1:3] *= -1

        all_trajectories[f"trajectories_{i}"] = normalized_trajs.astype(np.float32)
        all_metas.append(meta)

    # --- 3. 포인트 클라우드 처리 ---
    pcd_file = os.path.join(colmap_0_dir, "sparse", "0", "points3D.ply")
    # (pcd 파일 생성 로직은 기존과 동일)
    if not os.path.exists(pcd_file):
        try:
            cameras_extrinsic_file = os.path.join(colmap_0_dir, "sparse/0", "images.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(colmap_0_dir, "sparse/0", "images.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        nerf_normalization = getNerfppNorm(cam_extrinsics)
        print("Converting point3d.bin to .ply...")
        try:
            bin_path = os.path.join(colmap_0_dir, "sparse", "0", "points3D.bin")
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            txt_path = os.path.join(colmap_0_dir, "sparse", "0", "points3D.txt")
            xyz, rgb, _ = read_points3D_text(txt_path)
        xyz = xyz / nerf_normalization['radius']
        storePly(pcd_file, xyz, rgb)

    normalized_pcd_points = None
    pcd_colors = None
    if os.path.exists(pcd_file):
        pcd_points_world, pcd_colors_orig = load_ply(pcd_file)
        
        # COLMAP 포인트 클라우드를 기준 좌표계로 변환
        homogeneous_pcd = np.concatenate([pcd_points_world, np.ones((pcd_points_world.shape[0], 1))], axis=1)
        transformed_pcd = (world_to_reference_matrix @ homogeneous_pcd.T).T
        normalized_pcd_points = transformed_pcd[:, :3].astype(np.float32)
        
        # 시각화 라이브러리 좌표계에 맞춤 (Y, Z 축 반전)
        normalized_pcd_points[:, 1:3] *= -1
        
        # 포인트 클라우드 색상을 회색으로 통일
        pcd_colors = pcd_colors_orig.astype(np.float32)
        pcd_colors[:, 0:3] = np.array([0.5, 0.5, 0.5])

    # --- 4. 데이터 취합 및 압축 ---
    combined_arrays = {
        **all_trajectories,
        "pcd_points": normalized_pcd_points,
        "pcd_colors": pcd_colors
    }

    header = {}
    blob_parts = []
    offset = 0
    for key, arr in combined_arrays.items():
        if arr is not None:
            arr = np.ascontiguousarray(arr)
            arr_bytes = arr.tobytes()
            header[key] = {"dtype": str(arr.dtype), "shape": arr.shape, "offset": offset, "length": len(arr_bytes)}
            blob_parts.append(arr_bytes)
            offset += len(arr_bytes)
    
    raw_blob = b"".join(blob_parts)
    compressed_blob = zlib.compress(raw_blob, level=1)
    
    header["meta"] = {
        "baseFrameRate": fps,
        "trajectory_metas": all_metas, # 메타데이터를 리스트로 저장
    }
    
    compress_and_write(output_file, header, compressed_blob)
    
    if static_html_file is not None:
        with open(output_file, "rb") as f:
            encoded_blob = base64.b64encode(f.read()).decode("ascii")
        with open(viz_html_path, "r", encoding="utf-8") as f:
            html_template = f.read()
        injected_html = html_template.replace(
            "<head>",
            f"<head>\n<script>window.embeddedBase64 = `{encoded_blob}`;</script>"
        )
        with open(static_html_file, "w", encoding="utf-8") as f:
            f.write(injected_html)
# ====================================================================
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ====================================================================


def main():
    parser = argparse.ArgumentParser(description="Visualize N-camera 3D trajectories with a static point cloud.")
    # ▼▼▼ 여러 개의 .npz 파일을 받도록 수정 ▼▼▼
    parser.add_argument('input_files', nargs='+', help='Paths to the input .result.npz files. The first file is used as the reference coordinate system.')
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    parser.add_argument('--pcd', type=str, required=True, help='Path to the COLMAP directory for point cloud')
    parser.add_argument('--fps', type=int, default=10, help='Base frame rate for playback')
    parser.add_argument('--port', '-p', type=int, default=DEFAULT_PORT, help=f'Port to serve the visualization (default: {DEFAULT_PORT})')
    parser.add_argument('--static-html', '-s', type=str, default=None, help='Path to output a single, self-contained static HTML file')
    
    args = parser.parse_args()
    
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        process_data_with_pcd(
            args.input_files,
            args.pcd,
            temp_path / "data.bin",
            args.static_html,
            fps=args.fps
        )
        
        if args.static_html is not None:
            print(f"Static HTML file created at: {args.static_html}")
            return
            
        shutil.copy(viz_html_path, temp_path / "index.html")
        os.chdir(temp_path)
        host = "0.0.0.0"
        port = int(args.port)
        Handler = NoFaviconHandler
        
        with ThreadingTCPServer((host, port), Handler) as httpd:
            hostname = socket.gethostname()
            try:
                local_ip = socket.gethostbyname(hostname)
            except socket.gaierror:
                local_ip = "127.0.0.1"
            print(f"Serving at http://localhost:{port} or http://{local_ip}:{port}")
            print("Open this URL in your browser. Press Ctrl+C to stop.")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")

if __name__ == "__main__":
    main()