from isaacgym import gymapi
import numpy as np

# Print all sim params and its value
def recur_class_print(obj, indent=0):
    # if indent > 100:
    #     print(" " * indent + f"...")
    #     return
    for attr in dir(obj):
        if not attr.startswith("_"):
            value = getattr(obj, attr)
            if isinstance(value, (int, float, str, bool, np.dtype)):
                print(" " * indent + f"{attr}: {value}")
            elif isinstance(value, list):
                print(" " * indent + f"{attr}: [list of {len(value)} items]")
            else:
                # print(type(value))
                # print(" " * indent + f"{attr}: {value}")
                recur_class_print(value, indent+1)

def recur_class_print2(obj):
    """Print all attributes of an object without recursion."""
    stack = [(obj, 0)]  # Stack to hold objects and their indentation levels
    
    while stack:
        current_obj, indent = stack.pop()
        for attr in dir(current_obj):
            if not attr.startswith("_"):
                value = getattr(current_obj, attr)
                if isinstance(value, (int, float, str, bool, np.dtype)):
                    print(" " * indent + f"{attr}: {value}")
                elif isinstance(value, list):
                    print(" " * indent + f"{attr}: [list of {len(value)} items]")
                elif not isinstance(value, (type, type(None))):  # Avoid printing types or None
                    stack.append((value, indent + 1))



sim_param = gymapi.SimParams()
recur_class_print2(gymapi.ContactCollection(0))
raise ValueError("check the code above")