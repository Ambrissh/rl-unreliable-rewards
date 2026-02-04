import torch
import os

class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, agent, episode, episode_durations, filename='latest.pth'):
        """Save training checkpoint"""
        checkpoint = {
            'episode': episode,
            'policy_net': agent.policy_net.state_dict(),
            'target_net': agent.target_net.state_dict(),
            'optimizer': agent.optimizer.state_dict(),
            'steps_done': agent.steps_done,
            'episode_durations': episode_durations,
        }
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load(self, agent, filename='latest.pth'):
        """Load training checkpoint"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"No checkpoint found at {filepath}")
            return 0
        
        checkpoint = torch.load(filepath, map_location=agent.device)
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.steps_done = checkpoint['steps_done']
        
        print(f"Checkpoint loaded from episode {checkpoint['episode']}")
        return checkpoint['episode'], checkpoint.get('episode_durations', [])
    
    def list_checkpoints(self):
        """List all available checkpoints"""
        files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        return sorted(files)
