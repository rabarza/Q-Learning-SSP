import numpy as np

class LogNormalDistribution:
    """ Si X es una variable aleatoria con distribución normal, entonces Y = e^X distribuye log-normal. """
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
        return np.exp(self.mean + (self.std ** 2) / 2)
    def __str__(self) -> str:
        return f"LogNormalDistribution(mean={self.mean}, std={self.std})"

class NormalDistribution:
    """ Implementación de una distribución normal """
    def __init__(self, mean, std=1.0):
        self.mean = mean
        self.std = std
    def set_parameters(self, mean, std):
        self.mean = mean
        self.std = std
    def get_sample(self):
        sample = np.random.normal(self.mean, self.std)
        while sample <= 0:
            sample = np.random.normal(self.mean, self.std)
        return sample
    def get_expectation(self):
        return self.mean
    def __str__(self) -> str:
        return f"NormalDistribution(mean={self.mean}, std={self.std})"

class UniformDistribution:
    """ Implementación de una distribución uniforme """
    def __init__(self, low, high):
        self.low = low
        self.high = high
    def set_parameters(self, low, high):
        self.low = low
        self.high = high
    def get_sample(self):
        sample = np.random.uniform(self.low, self.high)
        while sample <= 0:
            sample = np.random.uniform(self.low, self.high)
        return sample
    def get_expectation(self):
        return (self.low + self.high) / 2
    def __str__(self) -> str:
        return f"UniformDistribution(low={self.low}, high={self.high})"

def validate_positive_parameters(arc_length, avg_speed):
    if arc_length < 0 or avg_speed <= 0:
        raise ValueError("Both arc_length and avg_speed must be greater than zero.")

def lognormal_expected_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)
    
    mu_x = avg_speed
    sigma_x = 6
    
    mu_speed = np.log((mu_x ** 2) / np.sqrt((mu_x ** 2 + sigma_x ** 2)))
    sigma_speed = np.sqrt(np.log((sigma_x ** 2 / (mu_x ** 2)) + 1)) 
    
    time_distr = LogNormalDistribution(-mu_speed, sigma_speed)
    time = arc_length * time_distr.get_expectation()
    return time

def lognormal_random_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)
    
    mu_x = avg_speed
    sigma_x = 6
    
    mu_speed = np.log((mu_x ** 2) / np.sqrt((mu_x ** 2 + sigma_x ** 2)))
    sigma_speed = np.sqrt(np.log((sigma_x ** 2 / (mu_x ** 2)) + 1)) 

    time_distr = LogNormalDistribution(-mu_speed, sigma_speed)
    time = arc_length * time_distr.get_sample()
    return time

def lognormal_random_speed(avg_speed=25):
    validate_positive_parameters(1, avg_speed)  # arc_length is irrelevant here but included for consistency
    
    mu_x = avg_speed
    sigma_x = 6
    
    mu_speed = np.log((mu_x ** 2) / np.sqrt((mu_x ** 2 + sigma_x ** 2)))
    sigma_speed = np.sqrt(np.log((sigma_x ** 2 / (mu_x ** 2)) + 1)) 
    
    speed = LogNormalDistribution(mu_speed, sigma_speed).get_sample()
    return speed

def normal_expected_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)
    
    mean = avg_speed
    std = 6
    
    time_distr = NormalDistribution(mean, std)
    time = arc_length / time_distr.get_expectation()
    return time

def normal_random_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)
    
    mean = avg_speed
    std = 6
    
    time_distr = NormalDistribution(mean, std)
    time = arc_length / time_distr.get_sample()
    return time

def normal_random_speed(avg_speed=25):
    validate_positive_parameters(1, avg_speed)  # arc_length is irrelevant here but included for consistency
    
    mean = avg_speed
    std = 6
    
    speed = NormalDistribution(mean, std).get_sample()
    return speed

def uniform_expected_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)
    
    low = max(avg_speed - 6, 0)
    high = avg_speed + 6
    
    time_distr = UniformDistribution(low, high)
    time = arc_length / time_distr.get_expectation()
    return time

def uniform_random_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)
    
    low = max(avg_speed - 6, 0)
    high = avg_speed + 6
    
    time_distr = UniformDistribution(low, high)
    time = arc_length / time_distr.get_sample()
    return time

def uniform_random_speed(avg_speed=25):
    validate_positive_parameters(1, avg_speed)  # arc_length is irrelevant here but included for consistency
    
    low = max(avg_speed - 6, 0)
    high = avg_speed + 6
    
    speed = UniformDistribution(low, high).get_sample()
    return speed

def expected_time(arc_length, avg_speed=25, distribution="lognormal"):
    validate_positive_parameters(arc_length, avg_speed)
    
    if distribution == "lognormal":
        return lognormal_expected_time(arc_length, avg_speed)
    elif distribution == "normal":
        return normal_expected_time(arc_length, avg_speed)
    elif distribution == "uniform":
        return uniform_expected_time(arc_length, avg_speed)
    else:
        raise ValueError("Unsupported distribution type")

def random_time(arc_length, avg_speed=25, distribution="lognormal"):
    """Returns a random travel time given an arc length and average speed. The distribution type can be lognormal, normal, or uniform."""
    
    validate_positive_parameters(arc_length, avg_speed)
    
    if distribution == "lognormal":
        return lognormal_random_time(arc_length, avg_speed)
    elif distribution == "normal":
        return normal_random_time(arc_length, avg_speed)
    elif distribution == "uniform":
        return uniform_random_time(arc_length, avg_speed)
    else:
        raise ValueError("Unsupported distribution type")

def random_speed(avg_speed=25, distribution="lognormal"):
    validate_positive_parameters(1, avg_speed)  # arc_length is irrelevant here but included for consistency
    
    if distribution == "lognormal":
        return lognormal_random_speed(avg_speed)
    elif distribution == "normal":
        return normal_random_speed(avg_speed)
    elif distribution == "uniform":
        return uniform_random_speed(avg_speed)
    else:
        raise ValueError("Unsupported distribution type")

if __name__ == "__main__":
    arc_length = 100 # km
    avg_speed = 25 # km/h
    
    # Test lognormal distribution
    print("Lognormal Expected Time:", expected_time(arc_length, avg_speed, "lognormal"))
    print("Lognormal Random Time:", random_time(arc_length, avg_speed, "lognormal"))
    print("Lognormal Random Speed:", random_speed(avg_speed, "lognormal"))
    
    # Test normal distribution
    print("Normal Expected Time:", expected_time(arc_length, avg_speed, "normal"))
    print("Normal Random Time:", random_time(arc_length, avg_speed, "normal"))
    print("Normal Random Speed:", random_speed(avg_speed, "normal"))
    
    # Test uniform distribution
    print("Uniform Expected Time:", expected_time(arc_length, avg_speed, "uniform"))
    print("Uniform Random Time:", random_time(arc_length, avg_speed, "uniform"))
    print("Uniform Random Speed:", random_speed(avg_speed, "uniform"))
