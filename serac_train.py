from easyeditor import EditTrainer, SERACTrainingHparams, ZsreDataset
import os

if __name__ == "__main__":
    training_hparams = SERACTrainingHparams.from_hparams('./hparams/TRAINING/SERAC/llama-7b.yaml')
    train_ds = ZsreDataset('./data/zsre/zsre_mend_train_10000.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
    # train_ds = ZsreDataset('./data/tofu_train_dummy_zsre.json', config=training_hparams)
    # eval_ds = ZsreDataset('./data/tofu_test_dummy_zsre.jsonn', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()

    # model_save_dir = './outputs/SERAC_zsre_train'
    # os.makedirs(model_save_dir, exist_ok=True)
    # trainer.model.save_pretrained(model_save_dir)