# Fichero: src/checkpoint.py
# MEJORA #9: Sistema de checkpointing para recuperación de errores

import json
import os
from typing import List, Tuple
from src.concepts import AlgorithmicConcept

class CheckpointManager:
    """
    Gestor de checkpoints para guardar y recuperar el estado de la evolución
    Permite reanudar desde el último punto guardado en caso de error
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self, 
        population: List[AlgorithmicConcept], 
        generation: int,
        problem_description: str = ""
    ):
        """
        Guarda un checkpoint de la población actual
        
        Args:
            population: Lista de conceptos actuales
            generation: Número de generación
            problem_description: Descripción del problema (opcional)
        """
        checkpoint_file = os.path.join(
            self.checkpoint_dir, 
            f"gen_{generation:03d}.json"
        )
        
        data = {
            'generation': generation,
            'problem_description': problem_description,
            'population_size': len(population),
            'population': [c.model_dump() for c in population]
        }
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"  💾 Checkpoint guardado: gen_{generation:03d}.json")
        except Exception as e:
            print(f"  ⚠️ Error al guardar checkpoint: {e}")
    
    def load_latest_checkpoint(self) -> Tuple[List[AlgorithmicConcept], int, str]:
        """
        Carga el checkpoint más reciente
        
        Returns:
            (population, generation, problem_description)
            Si no hay checkpoints, retorna ([], 0, "")
        """
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir) 
            if f.startswith('gen_') and f.endswith('.json')
        ]
        
        if not checkpoints:
            return [], 0, ""
        
        # Ordenar y tomar el más reciente
        latest = sorted(checkpoints)[-1]
        checkpoint_file = os.path.join(self.checkpoint_dir, latest)
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            population = [
                AlgorithmicConcept(**c) 
                for c in data.get('population', [])
            ]
            generation = data.get('generation', 0)
            problem_description = data.get('problem_description', "")
            
            print(f"  📂 Checkpoint cargado: {latest}")
            print(f"     Generación: {generation}")
            print(f"     Población: {len(population)} conceptos")
            
            return population, generation, problem_description
            
        except Exception as e:
            print(f"  ❌ Error al cargar checkpoint: {e}")
            return [], 0, ""
    
    def list_checkpoints(self) -> List[str]:
        """Lista todos los checkpoints disponibles"""
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('gen_') and f.endswith('.json')
        ]
        return sorted(checkpoints)
    
    def delete_old_checkpoints(self, keep_last_n: int = 5):
        """
        Elimina checkpoints antiguos, manteniendo solo los últimos N
        
        Args:
            keep_last_n: Número de checkpoints a mantener
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last_n:
            return
        
        to_delete = checkpoints[:-keep_last_n]
        
        for checkpoint in to_delete:
            try:
                os.remove(os.path.join(self.checkpoint_dir, checkpoint))
                print(f"  🗑️ Checkpoint antiguo eliminado: {checkpoint}")
            except Exception as e:
                print(f"  ⚠️ Error al eliminar {checkpoint}: {e}")