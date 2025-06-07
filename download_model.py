from modelscope import snapshot_download
import chromadb

model_dir = snapshot_download(
    model_id="BAAI/bge-large-zh-v1.5",
    cache_dir="./model_cache")

