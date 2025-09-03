import os
import sys
from config import Config
from src.trainer import Trainer
from src.utils import predict_category


def main():
    """Main function"""
    print("🏪 Vietnamese Receipt Classification System")
    print("📁 Dataset: viet_receipt_categorized_label.xlsx")
    print("🔧 Simple preprocessing + BoW/TF-IDF/Embeddings + GA-Voting")
    print("=" * 60)

    # Initialize configuration
    config = Config()

    # Check if dataset exists
    if not os.path.exists(config.EXCEL_FILE_PATH):
        print(f"❌ Dataset not found: {config.EXCEL_FILE_PATH}")
        print("Please ensure the Excel file is in the project root directory.")
        return

    try:
        # Run training pipeline
        print("🚀 Starting training pipeline...")
        trainer = Trainer(config)
        results = trainer.run_training()

        print(f"\n🎉 Training completed successfully!")
        print(f"💾 Model saved to: {config.MODEL_SAVE_PATH}")
        print(f"📊 Plots saved to: {config.PLOTS_DIR}/")

        # Demo predictions
        demo_predictions(config.MODEL_SAVE_PATH)

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


def demo_predictions(model_path: str):
    """Run demo predictions"""
    print(f"\n🔮 Demo Predictions:")

    demo_texts = [
        "Hoá đơn thanh toán tại Feel Coffee với Yogurt Very Berry giá 22.000 VND",
        "Hoá đơn VinCommerce sữa Vinamilk TTi giá 33.100 đồng",
        "Thanh toán VinID Pay tổng tiền 21.664.448 đồng mua sắm gia đình",
        "Hóa đơn thuốc paracetamol bệnh viện giá 15.000",
        "Hoá đơn siêu thị CP Giỏ bb 500g giá 111.200"
    ]

    for i, text in enumerate(demo_texts, 1):
        try:
            result = predict_category(text, model_path)
            print(f"\n   {i}. Text: {text[:50]}...")
            print(f"      → Category: {result['predicted_category']}")
            print(f"      → Confidence: {result['confidence']:.3f}")

            # Show top 3 if confidence is not very high
            if result['confidence'] < 0.8:
                print(f"      → Top 3:")
                for j, (cat, prob) in enumerate(result['top_3_predictions'][:3]):
                    print(f"         {j+1}. {cat}: {prob:.3f}")

        except Exception as e:
            print(f"      → Error: {e}")


if __name__ == "__main__":
    main()
