from PIL import Image
import torch
from network.network_builder import get_gazelle_model

model, transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
model.load_gazelle_state_dict(torch.load("gazelle_dinov2_vitl14_inout.pt", weights_only=True))
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image = Image.open("data/WALIexample0.png").convert("RGB")
input = {
    "images": transform(image).unsqueeze(dim=0).to(device),    # tensor of shape [1, 3, 448, 448]
    "bboxes": [[(0.1, 0.2, 0.5, 0.7)]]              # list of lists of bbox tuples
}
# if only 1 face
input["bboxes"] = [[None]]

with torch.no_grad():
    output = model(input)
predicted_heatmap = output["heatmap"][0][0]        # access prediction for first person in first image. Tensor of size [64, 64]
predicted_inout = output["inout"][0][0]            # in/out of frame score (1 = in frame) (output["inout"] will be None  for non-inout models)

print("Predicted in/out frame: ", predicted_inout)

import matplotlib.pyplot as plt
from network.utils import visualize_heatmap

viz = visualize_heatmap(image, predicted_heatmap)
plt.imshow(viz)
plt.show()
