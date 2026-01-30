# Prediction interface for Cog - SAM3D Multi-view
# https://cog.run/python

import os
import subprocess
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import List

from cog import BasePredictor, Input, Path as CogPath  # noqa: cog is available at runtime

class Predictor(BasePredictor):
    def setup(self) -> None:
        """
        Setup SAM3D environment.
        Hydra patching is done at Docker build time (see cog.yaml).
        Models should be pre-downloaded locally for fast setup:
          cd sam3d-multi
          huggingface-cli download --repo-type model --local-dir checkpoints/hf-download facebook/sam-3d-objects
          mv checkpoints/hf-download/checkpoints checkpoints/hf
          rm -rf checkpoints/hf-download
        """
        self.sam3d_dir = Path(__file__).parent
        self.model_tag = "hf"

        checkpoint_dir = self.sam3d_dir / "checkpoints" / self.model_tag
        if checkpoint_dir.exists():
            print(f"SAM3D checkpoints found at {checkpoint_dir}")
        else:
            # Fallback: download at runtime if not pre-downloaded
            print(f"Checkpoints not found, downloading from HuggingFace...")
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                subprocess.run(["huggingface-cli", "auth", "login", "--token", hf_token], check=False)

            download_dir = self.sam3d_dir / "checkpoints" / "hf-download"
            result = subprocess.run([
                "huggingface-cli", "download",
                "--repo-type", "model",
                "--local-dir", str(download_dir),
                "--max-workers", "1",
                "facebook/sam-3d-objects"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"Failed to download checkpoints: {result.stderr}")

            shutil.move(str(download_dir / "checkpoints"), str(checkpoint_dir))
            shutil.rmtree(download_dir, ignore_errors=True)
            print(f"Checkpoints downloaded to {checkpoint_dir}")

        print("SAM3D setup completed successfully")

    def predict(
        self,
        input_archive: CogPath = Input(
            description="ZIP or TAR archive containing images and masks. Images: X.png, Masks: X_mask.png"
        ),
        mask_prompt: str = Input(
            description="Mask folder name. If empty, images and masks are in the same directory. If specified, images are in input/images/, masks in input/{mask_prompt}/",
            default=""
        ),
        image_names: str = Input(
            description="Comma-separated image names (without extension), e.g., 'image1,view_a' or '1,2'. If empty, use all available images",
            default=""
        ),
        seed: int = Input(
            description="Random seed for reproducibility",
            default=42,
            ge=0,
            le=2147483647
        ),
        stage1_steps: int = Input(
            description="Stage 1 inference steps",
            default=50,
            ge=10,
            le=100
        ),
        stage2_steps: int = Input(
            description="Stage 2 inference steps",
            default=25,
            ge=10,
            le=100
        ),
        decode_formats: str = Input(
            description="Output formats, comma-separated: 'gaussian', 'mesh', or 'gaussian,mesh'",
            default="gaussian,mesh"
        ),
        model_tag: str = Input(
            description="Model checkpoint tag",
            default="hf"
        ),
        output_dir: str = Input(
            description="Output directory path. If specified, results are copied here. If empty, results stay in visualization/",
            default=""
        ),
    ) -> List[CogPath]:
        """
        Run SAM3D 3D reconstruction using run_inference.py subprocess
        """
        # Extract archive to temp directory
        work_dir = Path(tempfile.mkdtemp())
        input_path = work_dir / "input"
        input_path.mkdir()

        archive_path = str(input_archive)
        print(f"Extracting archive: {archive_path}")

        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(input_path)
        elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(input_path)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}. Use .zip, .tar, .tar.gz, or .tgz")

        # Check if files are in a subdirectory
        contents = list(input_path.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            input_path = contents[0]

        print(f"Extracted to: {input_path}")
        print(f"Contents: {list(input_path.iterdir())}")

        # Build command for run_inference.py
        cmd = [
            "python", "./run_inference.py",
            "--input_path", str(input_path),
            "--seed", str(seed),
            "--stage1_steps", str(stage1_steps),
            "--stage2_steps", str(stage2_steps),
            "--decode_formats", decode_formats,
            "--model_tag", model_tag,
        ]

        # Add optional parameters
        if mask_prompt:
            cmd.extend(["--mask_prompt", mask_prompt])

        if image_names:
            cmd.extend(["--image_names", image_names])

        print(f"Running SAM3D inference: {' '.join(cmd)}")

        # Run inference subprocess with real-time output
        process = subprocess.Popen(
            cmd,
            cwd=str(self.sam3d_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)

        process.wait()

        if process.returncode != 0:
            raise Exception(f"SAM3D inference failed with return code {process.returncode}")

        # Collect output files from visualization directory
        # Output is saved to visualization/{dir_name}/ based on input
        output_paths = []
        viz_dir = self.sam3d_dir / "visualization"

        if viz_dir.exists():
            # Find the most recently created output directory
            output_dirs = sorted(
                [d for d in viz_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            if output_dirs:
                latest_output = output_dirs[0]
                print(f"Collecting results from {latest_output}")

                # If output_dir is specified, copy files there
                if output_dir:
                    output_dir_path = Path(output_dir)
                    output_dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"Copying results to: {output_dir_path}")

                    for file in latest_output.glob("*"):
                        if file.is_file() and file.suffix in ['.glb', '.ply']:
                            dest_file = output_dir_path / file.name
                            shutil.copy2(file, dest_file)
                            output_paths.append(CogPath(dest_file))
                            print(f"  Copied: {file.name} -> {dest_file}")
                else:
                    # Return files from visualization directory
                    for file in latest_output.glob("*"):
                        if file.is_file() and file.suffix in ['.glb', '.ply']:
                            output_paths.append(CogPath(file))
                            print(f"  Found output: {file.name}")

        # Cleanup temp directory
        shutil.rmtree(work_dir, ignore_errors=True)

        if not output_paths:
            raise Exception("No output files generated from inference")

        return output_paths
