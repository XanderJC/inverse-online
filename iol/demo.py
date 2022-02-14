from iol.models import AdaptiveLinearModel
from iol.data_loading import generate_linear_dataset
from iol.constants import HYPERPARAMS


model = AdaptiveLinearModel(**HYPERPARAMS)

training_data = generate_linear_dataset(10000, 50, seed=41310)
validation_data = generate_linear_dataset(1000, 50, seed=41311).get_whole_batch()
test_data = generate_linear_dataset(1000, 50, seed=41312).get_whole_batch()

model.fit(
    training_data,
    batch_size=100,
    epochs=5,
    learning_rate=0.01,
    validation_set=validation_data,
)

print(model.validation(test_data))
