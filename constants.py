"""Constants for processing user csv data files."""

KITCHEN_CSV_KEYS = {
    "X": {
        "A": "top cabinet",
        "B": "top drawers",
        "C": "bottom cabinets",
        "D": "pantry cabinet",
        "E": "cabinets under the island kitchen",
        "F": "fridge",
        "G": "sink",
        "H": "kitchen counter",
        "I": "stovetop",
        "Oven": "oven",
    },
    "Y": {
        "A": "top cabinet",
        "B": "top drawers",
        "C": "bottom cabinets",
        "D": "pantry cabinet",
        "E": "fridge",
        "F": "sink",
        "G": "kitchen counter",
        "H": "stovetop",
        "Oven": "oven",
    },
    "Z": {
        "A": "top cabinet",
        "B": "top drawers",
        "C": "bottom cabinets",
        "D": "pantry cabinet",
        "E": "cabinets under the island kitchen",
        "F": "fridge",
        "G": "sink",
        "H": "kitchen counter",
        "I": "stovetop",
        "Oven": "oven",
    }
}

LIVING_ROOM_CSV_KEYS = {
    "X": {
        "A": "top of cabinet",  # TODO: change
        "B": "tv unit shelves",
        "C": "tv unit bottom cabinets",
        "D": "tv unit surface",
        "E": "tv unit bottom drawers",
        "F": "coffee table",
    },
    "Y": {
        "A": "tv unit surface",
        "B": "tv unit top shelves",
        "C": "tv unit bottom shelves",
        "D": "chair",
        "E": "coffee table",
    },
    "Z": {
        "A": "glass cabinet",
        "B": "side table",
        "C": "coffee table",
        "D": "drawer unit",
        "E": "ottoman"
    }
}

BATHROOM_CSV_KEYS = {
    "X": {
        "A": "shelf",
        "B": "counter",
        "C": "cabinets"
        },
    "Y": {
        "A": "drawers",
        "B": "counter",
        "C": "cabinets"
    },
    "Z": {
        "A": "cabinets",
        "B": "top shelf",
        "C": "drawers"
    }
}

HALLWAY_CSV_KEYS = {
    "X": {
        "A": "small shelves",
        "B": "book shelves",
        "C": "top cabinet",
        "D": "bottom cabinet",
    },
    "Y": {
        "A": "drawers",
        "B": "large cabinet",
        "C": "small cabinet",
        "D": "storage top surface"
    },
    "Z": {
        "A": "top clear cabinet",
        "B": "top shelves",
        "C": "storage top surface",
        "D": "bottom clear cabinet",
        "E": "bottom drawer"
    }
}

HOME_OFFICE_CSV_KEYS = {
    "X": {
        "A": "top shelf", "B": "table top surface", "C": "desk drawers"
        },
    "Y": {
        "A": "table top surface",
        "B": "middle desk drawer",
        "C": "side desk drawers"
    },
    "Z": {
        "A": "table top surface",
        "B": "desk drawers"
    }
}

BEDROOM_CSV_KEYS = {
    "X": {
        "A": "wardrobe organizer",
        "B": "small closet shelves",
        "C": "top hanger rod",
        "D": "bottom hanger rod",
    },
    "Y": {
        "A": "top dresser cabinets",
        "B": "cupboard",
        "C": "lower dresser drawers",
        "D": "nightstand"
    },
    "Z": {
        "A": "cupboard",
        "B": "small closet shelves",
        "C": "bottom dresser drawers",
        "D": "vanity table surface",
        "E": "vanity table drawer"
    }
}

GARAGE_CSV_KEYS = {
    "X": {
        "A": "tool shelf",
        "B": "peg board",
        "C": "workspace surface",
        "D": "bottom shelves",
    }
}

HOME_LAYOUTS = {
    "X": {
        "Kitchen": KITCHEN_CSV_KEYS["X"],
        "Living Room": LIVING_ROOM_CSV_KEYS["X"],
        "Bathroom": BATHROOM_CSV_KEYS["X"],
        "Hallway": HALLWAY_CSV_KEYS["X"],
        "Home Office": HOME_OFFICE_CSV_KEYS["X"],
        "Bedroom": BEDROOM_CSV_KEYS["X"],
        "Garage": GARAGE_CSV_KEYS["X"],
    },
    "Y": {
        "Kitchen": KITCHEN_CSV_KEYS["Y"],
        "Living Room": LIVING_ROOM_CSV_KEYS["Y"],
        "Bathroom": BATHROOM_CSV_KEYS["Y"],
        "Hallway": HALLWAY_CSV_KEYS["Y"],
        "Home Office": HOME_OFFICE_CSV_KEYS["Y"],
        "Bedroom": BEDROOM_CSV_KEYS["Y"],
        "Garage": GARAGE_CSV_KEYS["X"],
    },
    "Z": {
        "Kitchen": KITCHEN_CSV_KEYS["Z"],
        "Living Room": LIVING_ROOM_CSV_KEYS["Z"],
        "Bathroom": BATHROOM_CSV_KEYS["Z"],
        "Hallway": HALLWAY_CSV_KEYS["Z"],
        "Home Office": HOME_OFFICE_CSV_KEYS["Z"],
        "Bedroom": BEDROOM_CSV_KEYS["Z"],
        "Garage": GARAGE_CSV_KEYS["X"],

    },
    "XY": {
        "Kitchen": KITCHEN_CSV_KEYS["X"],
        "Living Room": LIVING_ROOM_CSV_KEYS["X"],
        "Bathroom": BATHROOM_CSV_KEYS["X"],
        "Hallway": HALLWAY_CSV_KEYS["Y"],
        "Home Office": HOME_OFFICE_CSV_KEYS["Y"],
        "Bedroom": BEDROOM_CSV_KEYS["Y"],
        "Garage": GARAGE_CSV_KEYS["X"],
    },
    "YZ":{
        "Kitchen": KITCHEN_CSV_KEYS["Y"],
        "Living Room": LIVING_ROOM_CSV_KEYS["Y"],
        "Bathroom": BATHROOM_CSV_KEYS["Y"],
        "Hallway": HALLWAY_CSV_KEYS["Z"],
        "Home Office": HOME_OFFICE_CSV_KEYS["Z"],
        "Bedroom": BEDROOM_CSV_KEYS["Z"],
        "Garage": GARAGE_CSV_KEYS["X"],
    },
    "XZ": {
        "Kitchen": KITCHEN_CSV_KEYS["X"],
        "Living Room": LIVING_ROOM_CSV_KEYS["X"],
        "Bathroom": BATHROOM_CSV_KEYS["X"],
        "Hallway": HALLWAY_CSV_KEYS["Z"],
        "Home Office": HOME_OFFICE_CSV_KEYS["Z"],
        "Bedroom": BEDROOM_CSV_KEYS["Z"],
        "Garage": GARAGE_CSV_KEYS["X"],
    },
    "XYZ": {
        "Kitchen": KITCHEN_CSV_KEYS["X"],
        "Living Room": LIVING_ROOM_CSV_KEYS["X"],
        "Bathroom": BATHROOM_CSV_KEYS["Y"],
        "Hallway": HALLWAY_CSV_KEYS["Y"],
        "Home Office": HOME_OFFICE_CSV_KEYS["Z"],
        "Bedroom": BEDROOM_CSV_KEYS["Z"],
        "Garage": GARAGE_CSV_KEYS["X"],
    }
}

MATCHING_SURFACES_BY_LAYOUT = {
    "Kitchen": [
        ["A", "A", "A"],
        ["B", "B", "B"],
        ["C", "C", "C"],
        ["D", "D", "D"],
        ["F", "E", "F"],
        ["G", "F", "G"],
        ["H", "G", "H"],
        ["I", "H", "I"],
        ["Oven", "Oven", "Oven"],
        ],
    "Living Room": [
        ["B", "B", None],
        ["F", "E", "C"],
        ["D", "A", None],
    ],
    "Bathroom": [
        ["C", "C", "C"],
        ["B", "B", None],
    ],
    "Hallway": [
        ["D", "B", None],
        [None, "D", "C"],
        [None, "A", "E"],
        ["B", None, "B"],
    ],
    "Home Office": [
        ["B", "A", "A"],
        ["C", "B", None],

    ],
    "Bedroom": [],
}

LAYOUT_TO_INDEX = {
    "X": {
        "Kitchen": 0,
        "Living Room": 0,
        "Bathroom": 0,
        "Hallway": 0,
        "Home Office": 0,
        "Bedroom": 0,
        "Garage": 0,
    },
    "Y": {
        "Kitchen": 1,
        "Living Room": 1,
        "Bathroom": 1,
        "Hallway": 1,
        "Home Office": 1,
        "Bedroom": 1,
        "Garage": 1,
    },
    "Z": {
        "Kitchen": 2,
        "Living Room": 2,
        "Bathroom": 2,
        "Hallway": 2,
        "Home Office": 2,
        "Bedroom": 2,
        "Garage": 2,
    },
    "XY": {
        "Kitchen": 0,
        "Living Room": 0,
        "Bathroom": 0,
        "Hallway": 1,
        "Home Office": 1,
        "Bedroom": 2,
        "Garage": 0,
    },
    "YZ": {
        "Kitchen": 1,
        "Living Room": 1,
        "Bathroom": 1,
        "Hallway": 2,
        "Home Office": 2,
        "Bedroom": 2,
        "Garage": 0,
    },
    "XZ": {
        "Kitchen": 0,
        "Living Room": 0,
        "Bathroom": 0,
        "Hallway": 2,
        "Home Office": 2,
        "Bedroom": 2,
        "Garage": 0,
    },
    "XYZ": {
        "Kitchen": 0,
        "Living Room": 0,
        "Bathroom": 1,
        "Hallway": 1,
        "Home Office": 2,
        "Bedroom": 2,
        "Garage": 0,
    },
}
