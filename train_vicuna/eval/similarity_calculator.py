from sentence_transformers import SentenceTransformer
import numpy
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

class SimilaritiesExtractor:
       
    def __init__(self, model_name: str = 'all-MiniLM-L12-v2'):
        self.__model = SentenceTransformer(model_name)

    def get_sentence_transformer(self) -> SentenceTransformer:
        return self.__model
   
    def get_embeddings(self, sentences: List[str]) -> numpy.ndarray:
        return self.__model.encode(sentences)

    def get_list_similarities(self, references: List[str],
                              predictions: List[str]) -> numpy.ndarray:
        """
        Returns a list containing the cosine_similarities
        between 2 lists by index
        Ex: the similarities[0] is the cosine similarity between
        sentences1[0] and sentences2[0]
        """
        if len(references) != len(predictions):
            raise ValueError("Lists to compare must contain the same size")
        ref_embds = self.get_embeddings(references)
        pred_embds = self.get_embeddings(predictions)
        similarities = [float(cosine_similarity([ref_embds[i]], [pred_embds[i]])[0][0])
                        for i in range(len(ref_embds))]
        return similarities