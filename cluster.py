import os
import csv
import cv2
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== CONFIG ==========
MODEL_PATH = "checkpoints/checkpoint_epoch_50.pth"  # Your trained model
DATA_DIR = "dataset"
IMG_SIZE = 256
FRAME_STACK = 4
BATCH_SIZE = 8
NUM_CLUSTERS = 4  # Number of patterns to find

# ========== YOUR MODEL (copy from train.py) ==========
class ControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 9)

    def forward(self, x):
        features = self.cnn(x).flatten(1)
        controls = self.fc(features)
        return controls  # Your original model only returns controls
    
    def get_features(self, x):
        # NEW METHOD to extract features
        return self.cnn(x).flatten(1)
        controls = self.fc(features)
        return controls, features  # Return both controls and features



# ========== DATASET (copy from train.py) ==========
def find_runs(root):
    return [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if d.startswith("run") and os.path.isdir(os.path.join(root, d))
    ]


class ControlDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        runs = find_runs(root)
        
        for run in runs:
            frames_dir = os.path.join(run, "frames")
            csv_path = os.path.join(run, "actions.csv")
            if not os.path.exists(frames_dir) or not os.path.exists(csv_path):
                continue

            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))

            for i in range(FRAME_STACK - 1, len(rows)):
                self.samples.append((frames_dir, rows, i))

        if not self.samples:
            raise RuntimeError("Dataset is empty")

        print(f"Dataset samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_dir, rows, i = self.samples[idx]

        imgs = []
        for j in range(i - FRAME_STACK + 1, i + 1):
            img = cv2.imread(os.path.join(frames_dir, rows[j]["frame"]))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            imgs.append(np.transpose(img, (2, 0, 1)))

        x = np.concatenate(imgs, axis=0)  # 12×256×256

        y = np.array([
            float(rows[i]["w_s"]),
            float(rows[i]["a_d"]),
            float(rows[i]["q_e"]),
            float(rows[i]["space"]),
            float(rows[i]["shift_ctrl"]),
            float(rows[i]["mouse_dx"]),
            float(rows[i]["mouse_dy"]),
            float(rows[i]["left_click"]),
            float(rows[i]["right_click"])
        ], dtype=np.float32)

        return torch.tensor(x), torch.tensor(y)


# ========== EXTRACT FEATURES ==========
def extract_features(model, loader, device):
    """Extract CNN features and actions from all samples"""
    model.eval()
    
    all_features = []
    all_actions = []
    
    print("Extracting features from dataset...")
    with torch.no_grad():
        for imgs, actions in tqdm(loader):
            imgs = imgs.to(device)
            features = model.get_features(imgs)  # Use new method
            
            all_features.append(features.cpu().numpy())
            all_actions.append(actions.cpu().numpy())
    
    features = np.concatenate(all_features, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    
    print(f"Extracted {len(features)} feature vectors")
    return features, actions


# ========== CLUSTER ANALYSIS ==========
def analyze_clusters(features, actions, n_clusters=8):
    """Cluster the features and analyze patterns"""
    print(f"\nClustering into {n_clusters} patterns...")
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS - GAMEPLAY PATTERNS")
    print("="*60)
    
    # Analyze each cluster
    for i in range(n_clusters):
        mask = cluster_labels == i
        cluster_actions = actions[mask]
        
        if len(cluster_actions) == 0:
            continue
        
        # Calculate statistics
        count = mask.sum()
        pct = (count / len(actions)) * 100
        
        avg_ws = cluster_actions[:, 0].mean()
        avg_ad = cluster_actions[:, 1].mean()
        avg_qe = cluster_actions[:, 2].mean()
        avg_space = cluster_actions[:, 3].mean()
        avg_shift_ctrl = cluster_actions[:, 4].mean()
        avg_mouse_x = cluster_actions[:, 5].mean()
        avg_mouse_y = cluster_actions[:, 6].mean()
        avg_left = cluster_actions[:, 7].mean()
        avg_right = cluster_actions[:, 8].mean()
        
        print(f"\nCluster {i}: {count} samples ({pct:.1f}%)")
        print(f"  Movement:")
        print(f"    W/S axis: {avg_ws:+.3f} {'[W pressed]' if avg_ws > 0.3 else '[S pressed]' if avg_ws < -0.3 else ''}")
        print(f"    A/D axis: {avg_ad:+.3f} {'[D pressed]' if avg_ad > 0.3 else '[A pressed]' if avg_ad < -0.3 else ''}")
        print(f"    Q/E axis: {avg_qe:+.3f}")
        print(f"    Space: {avg_space:.3f}")
        print(f"    Shift/Ctrl: {avg_shift_ctrl:+.3f}")
        print(f"  Mouse:")
        print(f"    X movement: {avg_mouse_x:+.3f}")
        print(f"    Y movement: {avg_mouse_y:+.3f}")
        print(f"  Clicks:")
        print(f"    Left: {avg_left:.3f}, Right: {avg_right:.3f}")
        
        # Identify pattern
        pattern = identify_pattern(avg_ws, avg_ad, avg_mouse_x, avg_mouse_y, 
                                   avg_left, avg_right, avg_space)
        print(f"  Pattern: {pattern}")
    
    return cluster_labels, kmeans


def identify_pattern(ws, ad, mx, my, left, right, space):
    """Identify what gameplay pattern this represents"""
    patterns = []
    
    if abs(ws) > 0.3:
        patterns.append("Moving " + ("forward" if ws > 0 else "backward"))
    if abs(ad) > 0.3:
        patterns.append("Strafing " + ("right" if ad > 0 else "left"))
    if abs(mx) > 0.2:
        patterns.append("Turning " + ("right" if mx > 0 else "left"))
    if abs(my) > 0.2:
        patterns.append("Looking " + ("down" if my > 0 else "up"))
    if left > 0.3:
        patterns.append("Left clicking")
    if right > 0.3:
        patterns.append("Right clicking")
    if space > 0.3:
        patterns.append("Jumping")
    
    if not patterns:
        return "Idle/Standing"
    
    return ", ".join(patterns)


# ========== VISUALIZATION ==========
def visualize_clusters(features, cluster_labels, n_clusters):
    """Visualize clusters in 2D using PCA"""
    print("\nGenerating visualization...")
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=10)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Gameplay Patterns - Cluster Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cluster_visualization.png', dpi=150)
    print("Saved visualization to: cluster_visualization.png")
    plt.show()


# ========== MAIN ==========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    model = ControlNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    print("Model loaded successfully!\n")
    
    # Load dataset
    dataset = ControlDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Extract features
    features, actions = extract_features(model, loader, device)
    
    # Cluster and analyze
    cluster_labels, kmeans = analyze_clusters(features, actions, NUM_CLUSTERS)
    
    # Visualize
    visualize_clusters(features, cluster_labels, NUM_CLUSTERS)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
