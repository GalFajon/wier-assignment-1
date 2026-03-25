import json
import os
from pathlib import Path

import json

from Explaination import *


if __name__ == "__main__":

    with open("cluster_keywords.json", "r", encoding="utf-8") as f:
        cluster_keywords = json.load(f)

    # Generate descriptors
    descriptors = generate_cluster_tags(cluster_keywords)

    # Save to new JSON file
    with open("cluster_descriptors.json", "w", encoding="utf-8") as f:
        json.dump(descriptors, f, ensure_ascii=False, indent=2)