import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE

def train_tft(data):
    data = data.reset_index()
    data['time_idx'] = range(len(data))
    data['group'] = 0

    dataset = TimeSeriesDataSet(
        data,
        time_idx='time_idx',
        target='Close',
        group_ids=['group'],
        max_encoder_length=30,
        max_prediction_length=7,
        time_varying_known_reals=['time_idx'],
        time_varying_unknown_reals=['Close'],
        target_normalizer=NaNLabelEncoder()
    )

    dataloader = dataset.to_dataloader(train=True, batch_size=32, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        loss=SMAPE()
    )

    trainer = pl.Trainer(max_epochs=10, gpus=0)
    trainer.fit(model, dataloader)

    return model
