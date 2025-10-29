# Fichero: src/vector_index.py
# MEJORA #3: Indexación vectorial con FAISS para búsqueda eficiente

import faiss
import numpy as np
from typing import List, Tuple, Optional

class ConceptIndex:
    """
    Índice vectorial para búsqueda rápida de conceptos similares usando FAISS
    Reemplaza la búsqueda O(n) de cosine_similarity por búsqueda O(log n)
    """
    
    def __init__(self, dimension: int = 768):
        """
        Args:
            dimension: Dimensión de los embeddings (768 para text-embedding-004)
        """
        self.dimension = dimension
        # IndexFlatIP = Inner Product (equivalente a cosine similarity con vectores normalizados)
        self.index = faiss.IndexFlatIP(dimension)
        self.concept_ids: List[str] = []
        self.concept_map = {}  # id -> índice en FAISS
    
    def add_concept(self, concept_id: str, embedding: List[float]):
        """
        Añade un concepto al índice
        
        Args:
            concept_id: ID único del concepto
            embedding: Vector de embedding
        """
        if not embedding or len(embedding) != self.dimension:
            print(f"  ⚠️ Embedding inválido para concepto {concept_id}")
            return
        
        # Normalizar para que producto interno = similitud coseno
        embedding_np = np.array([embedding], dtype='float32')
        faiss.normalize_L2(embedding_np)
        
        # Añadir al índice
        idx = len(self.concept_ids)
        self.index.add(embedding_np)
        self.concept_ids.append(concept_id)
        self.concept_map[concept_id] = idx
    
    def find_similar(
        self, 
        embedding: List[float], 
        k: int = 5, 
        threshold: float = 0.95,
        exclude_id: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Encuentra los k conceptos más similares
        
        Args:
            embedding: Vector de embedding del concepto a comparar
            k: Número de resultados a devolver
            threshold: Umbral mínimo de similitud (0.0 a 1.0)
            exclude_id: ID de concepto a excluir (ej: el mismo concepto)
        
        Returns:
            Lista de (concept_id, similarity_score) ordenada por similitud desc
        """
        if not embedding or len(embedding) != self.dimension:
            return []
        
        if len(self.concept_ids) == 0:
            return []
        
        # Normalizar query
        embedding_np = np.array([embedding], dtype='float32')
        faiss.normalize_L2(embedding_np)
        
        # Buscar los k+1 más cercanos (por si excluimos uno)
        search_k = min(k + 1, len(self.concept_ids))
        distances, indices = self.index.search(embedding_np, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.concept_ids):
                continue
            
            concept_id = self.concept_ids[idx]
            similarity = float(dist)
            
            # Filtrar por threshold y exclude_id
            if similarity >= threshold and concept_id != exclude_id:
                results.append((concept_id, similarity))
        
        # Limitar a k resultados
        return results[:k]
    
    def get_all_embeddings_matrix(self) -> np.ndarray:
        """Devuelve matriz numpy con todos los embeddings (para compatibilidad)"""
        if len(self.concept_ids) == 0:
            return np.array([])
        
        # Reconstruir embeddings desde FAISS
        embeddings = faiss.rev_swig_ptr(self.index.xb, self.index.ntotal * self.dimension)
        return embeddings.reshape(self.index.ntotal, self.dimension)
    
    def size(self) -> int:
        """Retorna el número de conceptos indexados"""
        return len(self.concept_ids)