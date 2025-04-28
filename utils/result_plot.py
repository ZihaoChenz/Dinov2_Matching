import matplotlib.pyplot as plt
import os

class Plot:
    def __init__(self, num_epochs, train_losses, val_losses, checkpoint_folder, train_pos_similarities, train_neg_similarities, val_pos_similarities, val_neg_similarities):
        self.num_epochs = num_epochs
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.checkpoint_folder = checkpoint_folder
        self.train_pos_similarities = train_pos_similarities
        self.train_neg_similarities = train_neg_similarities
        self.val_pos_similarities = val_pos_similarities
        self.val_neg_similarities = val_neg_similarities
        # self.train_aucs = train_aucs
        # self.val_aucs = val_aucs
        # self.train_aps = train_aps
        # self.val_aps = val_aps
        # self.train_top_k_accs = train_top_k_accs
        # self.val_top_k_accs = val_top_k_accs

        self.plot()

    def plot(self):
        # Loss Curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_epochs + 1), self.train_losses, label="Train Loss")
        plt.plot(range(1, self.num_epochs + 1), self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curve")
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(self.checkpoint_folder, "loss_curve.png")
        plt.savefig(loss_plot_path)

        # Training Similarity
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_epochs + 1), self.train_pos_similarities, label="Avg Train Pos Similarity")
        plt.plot(range(1, self.num_epochs + 1), self.train_neg_similarities, label="Avg Train Neg Similarity")
        plt.xlabel("Epoch")
        plt.ylabel("Similarity")
        plt.title("Training Similarity")
        plt.legend()
        plt.grid(True)
        train_similarity_plot_path = os.path.join(self.checkpoint_folder, "train_similarity.png")
        plt.savefig(train_similarity_plot_path)

        # Validation Similarity
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_epochs + 1), self.val_pos_similarities, label="Avg Val Pos Similarity")
        plt.plot(range(1, self.num_epochs + 1), self.val_neg_similarities, label="Avg Val Neg Similarity")
        plt.xlabel("Epoch")
        plt.ylabel("Similarity")
        plt.title("Validation Similarity")
        plt.legend()
        plt.grid(True)
        val_similarity_plot_path = os.path.join(self.checkpoint_folder, "val_similarity.png")
        plt.savefig(val_similarity_plot_path)

        # # AUC-ROC Curve
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(1, self.num_epochs + 1), self.train_aucs, label="Train AUC-ROC")
        # plt.plot(range(1, self.num_epochs + 1), self.val_aucs, label="Validation AUC-ROC")
        # plt.xlabel("Epoch")
        # plt.ylabel("AUC-ROC")
        # plt.title("AUC-ROC Curve")
        # plt.legend()
        # plt.grid(True)
        # auc_plot_path = os.path.join(self.checkpoint_folder, "auc_curve.png")
        # plt.savefig(auc_plot_path)
        #
        # # Average Precision (AP) Curve
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(1, self.num_epochs + 1), self.train_aps, label="Train AP")
        # plt.plot(range(1, self.num_epochs + 1), self.val_aps, label="Validation AP")
        # plt.xlabel("Epoch")
        # plt.ylabel("AP")
        # plt.title("Average Precision Curve")
        # plt.legend()
        # plt.grid(True)
        # ap_plot_path = os.path.join(self.checkpoint_folder, "ap_curve.png")
        # plt.savefig(ap_plot_path)
        #
        # # Top-K Accuracy Curve
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(1, self.num_epochs + 1), self.train_top_k_accs, label="Train Top-K Accuracy")
        # plt.plot(range(1, self.num_epochs + 1), self.val_top_k_accs, label="Validation Top-K Accuracy")
        # plt.xlabel("Epoch")
        # plt.ylabel("Top-K Accuracy")
        # plt.title("Top-K Accuracy Curve")
        # plt.legend()
        # plt.grid(True)
        # top_k_acc_plot_path = os.path.join(self.checkpoint_folder, "top_k_accuracy_curve.png")
        # plt.savefig(top_k_acc_plot_path)
