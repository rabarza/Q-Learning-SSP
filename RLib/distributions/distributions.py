import numpy as np

class LogNormalDistribution:
    """ Si X es una variable aleatoria con distribución normal, entonces Y = e^X distribuye log-normal.
    """
    def __init__(self, n_mean, n_std=0.25):
        self.mean = n_mean
        self.std = n_std
    def set_parameters(self, n_mean, n_std):
        self.mean = n_mean
        self.std = n_std
    def get_sample(self):
        sample = np.random.lognormal(self.mean, self.std)
        return sample
    def get_sample_vector(self, size=1):
        return [self.get_sample() for i in range(size)]
    def get_expectation(self):
        return pow(np.e, self.mean + (self.std ** 2) / 2) 
    def __str__(self) -> str:
        return f"LogNormalDistribution(mean={self.mean}, std={self.std})"
    

def expected_speed(avg_speed=25):
    """
    Calcula la esperanza del inverso de la velocidad E(1/v) de un arco de acuerdo a la velocidad promedio. La esperanza de la distribución log-normal es e^(mu + sigma^2 / 2). Por lo tanto, si la velocidad tiene parametro mu, el inverso de la velocidad tiene parámetro -mu
    """                   
    mu_x = avg_speed
    sigma_x = 6
    
    mu_speed = np.log((mu_x ** 2) / np.sqrt((mu_x ** 2 + sigma_x ** 2)))
    sigma_speed = np.sqrt(np.log((sigma_x ** 2 / (mu_x ** 2)) + 1)) 
    
    speed = LogNormalDistribution(mu_speed, sigma_speed).get_expectation()
    return speed

def random_speed(avg_speed=25):
    """
    Calcula la velocidad aleatoria de un arco de acuerdo a la velocidad promedio. La esperanza de la distribución log-normal es e^(mu + sigma^2 / 2).
    """                   
    mu_x = avg_speed
    sigma_x = 6
    
    mu_speed = np.log((mu_x ** 2) / np.sqrt((mu_x ** 2 + sigma_x ** 2)))
    sigma_speed = np.sqrt(np.log((sigma_x ** 2 / (mu_x ** 2)) + 1)) 
    
    speed = LogNormalDistribution(mu_speed, sigma_speed).get_sample()
    return speed

def expected_time(arc_length, avg_speed=25):
    """
    Calcula el tiempo esperado de un arco de acuerdo a la distribución log-normal de la velocidad. Si la velocidad distribuye LogNormal con parámetros mu y std, el tiempo distribuye LogNormal con parámetros -mu y std. La esperanza de la distribución log-normal es e^(mu + sigma^2 / 2). 
    
    Para generar una distribución con una media mu_x y desviación estandar sigma_x deseadas, se utiliza mu = log(mu_x^2 / sqrt(mu_x^2 + sigma_x^2)) y std^2 = log(1 + sigma_x^2 / mu_x^2).
    """
    # Desired mean and standard deviation of logX
    mu_x = avg_speed
    sigma_x = 6
    
    mu_speed = np.log((mu_x ** 2) / np.sqrt((mu_x ** 2 + sigma_x ** 2)))
    sigma_speed = np.sqrt(np.log((sigma_x ** 2 / (mu_x ** 2)) + 1)) 
    
    time_distr = LogNormalDistribution(- mu_speed, sigma_speed)
    time = arc_length * time_distr.get_expectation()
    return time

def random_time(arc_length, avg_speed=25):
    """
    Calcula el tiempo aleatorio de un arco de acuerdo a la distribución log-normal del peso escogido ('weight').
    """
    mu_x = avg_speed
    sigma_x = 6
    
    mu_speed = np.log((mu_x ** 2) / np.sqrt((mu_x ** 2 + sigma_x ** 2)))
    sigma_speed = np.sqrt(np.log((sigma_x ** 2 / (mu_x ** 2)) + 1)) 

    time_distr = LogNormalDistribution(- mu_speed, sigma_speed)
    time = arc_length * time_distr.get_sample()
    # release object
    del time_distr
    return time