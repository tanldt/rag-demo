import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import rag_demo


class DummyModel:
    def __init__(self):
        self.encoded = False

    def encode(self, chunks, show_progress_bar=True):
        self.encoded = True
        return np.ones((len(chunks), 3))


class EmbedCacheTests(unittest.TestCase):
    def test_rebuilds_cache_when_chunk_count_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            np.savez(
                cache_dir / "embeddings.npz",
                embeddings=np.empty((0,)),
                chunks=np.array([], dtype=object),
            )
            model = DummyModel()

            with patch.object(rag_demo, "CACHE_DIR", cache_dir), patch.object(
                rag_demo, "get_model", return_value=model
            ):
                embeddings = rag_demo.embed_chunks(["one", "two"], rebuild=False)

        self.assertTrue(model.encoded)
        self.assertEqual(embeddings.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
