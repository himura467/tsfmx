FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
RUN apt update -y && \
  apt install -y wget git libarchive-tools && \
  wget -qO- https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN git clone https://github.com/himura467/tsfmx.git && \
  cd tsfmx && \
  ./scripts/clone_time_mmd.sh && \
  ./scripts/setup_mm_tsflib.sh && \
  uv sync --extra sweep --extra bench
WORKDIR /tsfmx
ENV PYTHONPATH=/tsfmx
RUN uv run python scripts/split_time_mmd_datasets.py \
  --train-ratio 0.7 \
  --val-ratio 0.1
