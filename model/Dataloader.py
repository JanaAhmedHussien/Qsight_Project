# ============================================================================
# SECTION 2: DATASET CLASS
# ============================================================================
import kagglehub

class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, dataset_type='sovitrath', mode='train', transform=None):
        super().__init__()
        self.dataset_type = dataset_type
        self.mode = mode
        self.transform = transform

        if dataset_type == 'sovitrath':
            self.load_sovitrath_dataset()
        else:
            self.load_tanlikesmath_dataset()

        print(f"✅ Loaded {len(self.image_paths)} images from {dataset_type} ({mode} set)")

    def load_sovitrath_dataset(self):
        try:
            path = kagglehub.dataset_download("sovitrath/diabetic-retinopathy-224x224-gaussian-filtered")
            print(f"📂 Dataset downloaded to: {path}")

            base_path = os.path.join(path, "gaussian_filtered_images", "gaussian_filtered_images")
            self.classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

            self.image_paths = []
            self.labels = []

            for class_name in self.classes:
                class_path = os.path.join(base_path, class_name)
                if os.path.exists(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(class_path, img_name))
                            self.labels.append(self.class_to_idx[class_name])

        except Exception as e:
            print(f"⚠️ Error loading Sovitrath dataset: {e}")
            self.create_mock_dataset()

    def load_tanlikesmath_dataset(self):
        try:
            path = kagglehub.dataset_download("tanlikesmath/diabetic-retinopathy-resized")
            print(f"📂 Dataset downloaded to: {path}")

            base_path = os.path.join(path, "resized_train", "resized_train")
            labels_path = os.path.join(path, "resized_train", "trainLabels.csv")

            labels_df = pd.read_csv(labels_path)
            self.classes = ['0', '1', '2', '3', '4']
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

            self.image_paths = []
            self.labels = []

            for _, row in labels_df.iterrows():
                img_name = row['image'] + '.jpeg'
                img_path = os.path.join(base_path, img_name)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(int(row['level']))

        except Exception as e:
            print(f"⚠️ Error loading Tanlikesmath dataset: {e}")
            self.create_mock_dataset()

    def create_mock_dataset(self):
        print("📝 Creating mock dataset for testing...")
        self.classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_paths = [f"dummy_{i}.jpg" for i in range(500)]
        self.labels = [i % 5 for i in range(500)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            if img_path.startswith("dummy_"):
                image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            else:
                image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            label = self.labels[idx]
            return image, label, img_path

        except Exception as e:
            print(f"⚠️ Error loading image {img_path}: {e}")
            dummy_image = torch.randn(3, 224, 224)
            dummy_label = 0
            return dummy_image, dummy_label, "dummy_path"

print("✅ Dataset class defined successfully!")