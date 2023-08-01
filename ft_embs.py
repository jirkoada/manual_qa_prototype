from typing import List
import fasttext

class FTembeddings:

    def __init__(self):
        self.model = fasttext.load_model("../models/cc.en.300.bin")
        
    def embed_documents(self, texts: List[str], chunk_size=0) -> List[List[float]]:
        return [self.model.get_sentence_vector(t.lower().replace('\n', ' ')) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.model.get_sentence_vector(text.lower().replace('\n', ' ')).tolist()