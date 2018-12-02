import matplotlib.pyplot as plt
import numpy as np

def Test_FillMissingValues():
    import normalize
    x1 = np.linspace(-6,6,10)
    y1 = np.sin(x1)
    x2 = np.linspace(-5,5,50)
    y2 = np.cos(x2)
    
    x3,y1new,y2new = normalize.FillMissingValues(x1,y1,x2,y2)
    
    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.scatter(x3,y1new,c="r")
    plt.scatter(x3,y2new,c="g")
    plt.show()
    
    quit()

if __name__ == "__main__":
    # Test_FillMissingValues()
    pass