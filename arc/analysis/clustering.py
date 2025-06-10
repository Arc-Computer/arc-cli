"""
Failure Clustering for Arc-Eval Production
Groups similar failures with clear, human-readable names to save debugging time
Production version adapted from experiments/src/analysis/failure_clustering.py
"""

import json
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import re


class FailureClusterer:
    """Clusters failures into clear, named patterns"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
    def extract_failures_from_results(self, evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract failures from evaluation results"""
        failures = []
        
        for result in evaluation_results.get('results', []):
            trajectory = result.get('trajectory', {})
            scenario = result.get('scenario', {})
            score = result.get('reliability_score', {})
            
            # Check for execution errors
            if trajectory.get('status') == 'error':
                failures.append({
                    'type': 'execution_error',
                    'scenario_name': scenario.get('name', 'Unknown'),
                    'task': scenario.get('task_prompt', ''),
                    'error_message': trajectory.get('error', 'Unknown error'),
                    'full_trajectory': trajectory,
                    'failure_text': f"Execution error: {trajectory.get('error', '')}"
                })
            
            # Check for low reliability scores
            elif score.get('overall_score', 100) < 70:
                # Find the lowest dimension
                dim_scores = score.get('dimension_scores', {})
                if dim_scores:
                    lowest_dim = min(dim_scores.items(), key=lambda x: x[1])
                    failures.append({
                        'type': 'low_reliability',
                        'scenario_name': scenario.get('name', 'Unknown'),
                        'task': scenario.get('task_prompt', ''),
                        'overall_score': score.get('overall_score', 0),
                        'lowest_dimension': lowest_dim[0],
                        'lowest_score': lowest_dim[1],
                        'full_trajectory': trajectory,
                        'failure_text': f"Low {lowest_dim[0]}: {lowest_dim[1]}% - {scenario.get('task_prompt', '')}"
                    })
            
            # Check for tool errors in trajectory
            for event in trajectory.get('full_trajectory', []):
                if event.get('type') == 'tool_call':
                    tool_output = str(event.get('tool_output', ''))
                    if 'error' in tool_output.lower():
                        failures.append({
                            'type': 'tool_error',
                            'scenario_name': scenario.get('name', 'Unknown'),
                            'task': scenario.get('task_prompt', ''),
                            'tool': event.get('tool', 'Unknown'),
                            'tool_input': event.get('tool_input', {}),
                            'error_output': tool_output,
                            'full_trajectory': trajectory,
                            'failure_text': f"Tool error in {event.get('tool', 'Unknown')}: {tool_output[:100]}"
                        })
        
        return failures
    
    def _generate_cluster_name(self, failures: List[Dict[str, Any]]) -> str:
        """Generate a clear, human-readable name for a cluster"""
        
        # Count failure types
        type_counts = defaultdict(int)
        tool_counts = defaultdict(int)
        dimension_counts = defaultdict(int)
        error_keywords = defaultdict(int)
        
        for failure in failures:
            type_counts[failure.get('type', 'unknown')] += 1
            
            if failure.get('type') == 'tool_error':
                tool_counts[failure.get('tool', 'unknown')] += 1
            elif failure.get('type') == 'low_reliability':
                dimension_counts[failure.get('lowest_dimension', 'unknown')] += 1
            
            # Extract error keywords
            error_text = failure.get('error_message', '') or failure.get('error_output', '')
            if error_text:
                # Look for common error patterns
                if 'timeout' in error_text.lower():
                    error_keywords['timeout'] += 1
                elif 'not found' in error_text.lower():
                    error_keywords['not_found'] += 1
                elif 'invalid' in error_text.lower():
                    error_keywords['invalid_input'] += 1
                elif 'connection' in error_text.lower():
                    error_keywords['connection'] += 1
        
        # Generate name based on most common patterns
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'unknown'
        
        if most_common_type == 'tool_error' and tool_counts:
            most_common_tool = max(tool_counts.items(), key=lambda x: x[1])[0]
            if error_keywords:
                most_common_error = max(error_keywords.items(), key=lambda x: x[1])[0]
                return f"{most_common_tool}_{most_common_error}"
            return f"{most_common_tool}_errors"
        
        elif most_common_type == 'low_reliability' and dimension_counts:
            most_common_dim = max(dimension_counts.items(), key=lambda x: x[1])[0]
            return f"low_{most_common_dim}_quality"
        
        elif most_common_type == 'execution_error' and error_keywords:
            most_common_error = max(error_keywords.items(), key=lambda x: x[1])[0]
            return f"execution_{most_common_error}"
        
        # Fallback
        return f"{most_common_type}_cluster"
    
    def cluster_failures(self, failures: List[Dict[str, Any]], 
                        min_cluster_size: int = 2) -> List[Dict[str, Any]]:
        """Cluster similar failures together"""
        
        if not failures:
            return []
        
        logging.info(f"Clustering {len(failures)} failures...")
        start_time = time.time()
        
        # Create feature vectors from failure text
        failure_texts = [f.get('failure_text', '') for f in failures]
        
        if len(failures) < 2:
            # Single failure, create one cluster
            return [{
                'id': 'cluster_single',
                'name': self._generate_cluster_name(failures),
                'failures': failures,
                'size': 1,
                'representative_failure': failures[0]
            }]
        
        # Vectorize failure texts
        try:
            feature_vectors = self.vectorizer.fit_transform(failure_texts)
        except (ValueError, TypeError):
            # If vectorization fails, use simple grouping by type
            return self._simple_clustering(failures)
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=0.3, min_samples=min_cluster_size, metric='cosine')
        cluster_labels = clustering.fit_predict(feature_vectors)
        
        # Group failures by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label == -1:
                # Noise point - create individual cluster
                clusters[f'outlier_{idx}'].append(failures[idx])
            else:
                clusters[f'cluster_{label}'].append(failures[idx])
        
        # Create cluster objects with clear names
        cluster_list = []
        for cluster_id, cluster_failures in clusters.items():
            cluster_name = self._generate_cluster_name(cluster_failures)
            
            # Find representative failure (closest to cluster center)
            if len(cluster_failures) > 1:
                cluster_indices = [i for i, label in enumerate(cluster_labels) 
                                 if label == (int(cluster_id.split('_')[1]) if 'cluster_' in cluster_id else -1)]
                if cluster_indices:
                    cluster_vectors = feature_vectors[cluster_indices]
                    centroid = cluster_vectors.mean(axis=0)
                    # Convert to numpy array to avoid sklearn matrix issues
                    if hasattr(centroid, 'A'):
                        centroid = np.asarray(centroid.A).reshape(1, -1)
                    else:
                        centroid = np.asarray(centroid).reshape(1, -1)
                    distances = cosine_similarity(cluster_vectors, centroid).flatten()
                    representative_idx = cluster_indices[distances.argmax()]
                    representative = failures[representative_idx]
                else:
                    representative = cluster_failures[0]
            else:
                representative = cluster_failures[0]
            
            cluster_list.append({
                'id': cluster_id,
                'name': cluster_name,
                'failures': cluster_failures,
                'size': len(cluster_failures),
                'representative_failure': representative
            })
        
        # Sort by cluster size
        cluster_list.sort(key=lambda x: x['size'], reverse=True)
        
        clustering_time = time.time() - start_time
        print(f"✓ Created {len(cluster_list)} clusters in {clustering_time:.2f}s")
        
        return cluster_list
    
    def _simple_clustering(self, failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple clustering by failure type when vectorization fails"""
        clusters = defaultdict(list)
        
        for failure in failures:
            key = f"{failure.get('type', 'unknown')}_{failure.get('tool', failure.get('lowest_dimension', 'general'))}"
            clusters[key].append(failure)
        
        cluster_list = []
        for key, cluster_failures in clusters.items():
            cluster_list.append({
                'id': key,
                'name': self._generate_cluster_name(cluster_failures),
                'failures': cluster_failures,
                'size': len(cluster_failures),
                'representative_failure': cluster_failures[0]
            })
        
        return sorted(cluster_list, key=lambda x: x['size'], reverse=True)
    
    def display_clusters(self, clusters: List[Dict[str, Any]]):
        """Display clusters in a simple format"""
        
        print("\n=== Failure Clusters ===")
        print(f"{'Cluster':<30} {'Size':>6} {'Representative Failure'}")
        print("-" * 80)
        
        for cluster in clusters[:10]:  # Show top 10
            rep_failure = cluster['representative_failure']
            failure_desc = rep_failure.get('error_message', '')[:40] or \
                          rep_failure.get('failure_text', '')[:40] or \
                          'Unknown failure'
            
            print(f"{cluster['name']:<30} {cluster['size']:>6} {failure_desc}")
        
        if len(clusters) > 10:
            print(f"\n... and {len(clusters) - 10} more clusters")
    
    def save_cluster_report(self, clusters: List[Dict[str, Any]], output_file: str = None):
        """Save detailed cluster report"""
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"failure_clusters_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_clusters': len(clusters),
            'total_failures': sum(c['size'] for c in clusters),
            'clusters': []
        }
        
        for cluster in clusters:
            cluster_data = {
                'name': cluster['name'],
                'size': cluster['size'],
                'representative': {
                    'scenario': cluster['representative_failure'].get('scenario_name', 'Unknown'),
                    'task': cluster['representative_failure'].get('task', ''),
                    'failure_type': cluster['representative_failure'].get('type', 'unknown')
                },
                'failure_summaries': []
            }
            
            # Add summaries of all failures in cluster
            for failure in cluster['failures'][:5]:  # First 5 failures
                cluster_data['failure_summaries'].append({
                    'scenario': failure.get('scenario_name', 'Unknown'),
                    'type': failure.get('type', 'unknown'),
                    'summary': failure.get('failure_text', '')[:100]
                })
            
            report['clusters'].append(cluster_data)
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Cluster report saved to {output_file}")
        
        return output_file