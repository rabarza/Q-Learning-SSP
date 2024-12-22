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
    if np.any(arc_length < 0) or np.any(avg_speed <= 0):
        raise ValueError(
            "Both arc_length and avg_speed must be greater than zero.")


def lognormal_expected_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)

    mu_x = avg_speed
    sigma_x = 6

    mu_speed = np.log((mu_x ** 2) / np.sqrt((mu_x ** 2 + sigma_x ** 2)))
    sigma_speed = np.sqrt(np.log((sigma_x ** 2 / (mu_x ** 2)) + 1))
    # Mean of 1 / X is exp(-mu + sigma^2 / 2)
    mean = np.exp(-mu_speed + (sigma_speed ** 2) / 2)
    time = arc_length * mean
    return time


def lognormal_random_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)

    mu_x = avg_speed
    sigma_x = 6

    mu_speed = np.log((mu_x ** 2) / np.sqrt((mu_x ** 2 + sigma_x ** 2)))
    sigma_speed = np.sqrt(np.log((sigma_x ** 2 / (mu_x ** 2)) + 1))
    # To sample 1 / X, where X is lognormal
    # it is equivalent to sample X with mean -mu and sigma
    # or equivalently sample X with mean mu and sigma
    # and then take the inverse
    random_speed_inverse = np.random.lognormal(-mu_speed, sigma_speed)
    time = arc_length * random_speed_inverse
    return time


def uniform_expected_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)

    low = max(avg_speed - 3, 0.001)
    high = avg_speed + 3

    inverse_mean = (np.log(high) - np.log(low)) / (high-low)
    time = arc_length * inverse_mean
    return time


def uniform_random_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)

    low = max(avg_speed - 3, 0.001)
    high = avg_speed + 3

    random_speed = np.random.uniform(low, high)
    time = arc_length / random_speed
    return time


def normal_expected_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)

    mean = avg_speed
    std = 6
    # the next step is false, the mean of 1 / X is not 1 / mean
    time = arc_length / mean
    return time


def normal_random_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)

    mean = avg_speed
    std = 6

    random_speed = np.random.normal(mean, std)
    time = arc_length / random_speed
    return time


def expected_time(arc_length, avg_speed=25, distribution="lognormal"):
    validate_positive_parameters(arc_length, avg_speed)

    if distribution == "lognormal":
        return lognormal_expected_time(arc_length, avg_speed) * 3.6
    elif distribution == "normal":
        return normal_expected_time(arc_length, avg_speed) * 3.6
    elif distribution == "uniform":
        return uniform_expected_time(arc_length, avg_speed) * 3.6
    else:
        raise ValueError("Unsupported distribution type")


def random_time(arc_length, avg_speed=25, distribution="lognormal"):
    """Returns a random travel time given an arc length and average speed. The distribution type can be lognormal, normal, or uniform. Arc length is assumed to be in meters and average speed in km/h. """
    validate_positive_parameters(arc_length, avg_speed)
    if distribution == "lognormal":
        return lognormal_random_time(arc_length, avg_speed) * 3.6
    elif distribution == "normal":
        return normal_random_time(arc_length, avg_speed) * 3.6
    elif distribution == "uniform":
        return uniform_random_time(arc_length, avg_speed) * 3.6
    else:
        raise ValueError("Unsupported distribution type")


if __name__ == "__main__":
    arc_length = 100  # km
    avg_speed = 25  # km/h

    # Test lognormal distribution
    print("Lognormal Expected Time:", expected_time(
        arc_length, avg_speed, "lognormal"))
    print("Lognormal Random Time:", random_time(
        arc_length, avg_speed, "lognormal"))

    # Test normal distribution
    print("Normal Expected Time:", expected_time(
        arc_length, avg_speed, "normal"))
    print("Normal Random Time:", random_time(arc_length, avg_speed, "normal"))

    # Test uniform distribution
    print("Uniform Expected Time:", expected_time(
        arc_length, avg_speed, "uniform"))
    print("Uniform Random Time:", random_time(
        arc_length, avg_speed, "uniform"))
