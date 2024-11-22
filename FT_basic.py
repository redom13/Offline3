import numpy as np
import matplotlib.pyplot as plt

def target_function(x, function_type):
    if function_type == "polynomial":
        return (x_values>=-2)*(x_values<=2)*x_values**2
    elif function_type == "triangular":
        return 2*np.abs(x/2.0-np.floor(x/2.0+0.5))
    elif function_type == "sawtooth":
        return x - np.floor(x)
    elif function_type == "rectangular":
        return np.where(np.cos(4*x)>=0,0.5,-0.5)*(x_values>=-2)*(x_values<=2)
    
# Fourier Transform 
def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)
    
    # Store the fourier transform results for each frequency. Handle the real and imaginary parts separately
    # use trapezoidal integration to calculate the real and imaginary parts of the FT
    for i, freq in enumerate(frequencies):
        # exp(-2πift) = cos(-2πft) - i*sin(-2πft)
        integrand_real = signal * np.cos(-2 * np.pi * freq * sampled_times)
        integrand_imag = signal * np.sin(-2 * np.pi * freq * sampled_times)
        
        ft_result_real[i] = np.trapz(integrand_real, sampled_times)
        ft_result_imag[i] = np.trapz(integrand_imag, sampled_times)

    return ft_result_real, ft_result_imag

# Inverse Fourier Transform 
def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    n = len(sampled_times)
    reconstructed_signal = np.zeros(n)
    # Reconstruct the signal by summing over all frequencies for each time in sampled_times.
    # use trapezoidal integration to calculate the real part
    # You have to return only the real part of the reconstructed signal
    ft_real, ft_imag = ft_signal
    
    for i, t in enumerate(sampled_times):
        # exp(2πift) = cos(2πft) + i*sin(2πft)
        integrand_real = (ft_real * np.cos(2*np.pi*frequencies*t) - 
                         ft_imag * np.sin(2*np.pi*frequencies*t))
        integrand_imag = (ft_real * np.sin(2*np.pi*frequencies*t) + 
                         ft_imag * np.cos(2*np.pi*frequencies*t))
        
        reconstructed_signal[i] = np.trapz(integrand_real, frequencies)

    return reconstructed_signal

# def triangular_wave(x):
#     return 2*np.abs(x/2.0-np.floor(x/2.0+0.5))

for function_type in ["rectangular"]:
    # Define the interval and function and generate appropriate x values and y values
    x_values = np.linspace(-10, 10, 1000)
    # y_values = (x_values>=-2)*(x_values<=2)*x_values**2
    # y_values = target_function(x_values, function_type)
    y_values = np.cos(x_values)

    # Plot the original function
    plt.figure(figsize=(12, 4))
    plt.plot(x_values, y_values, label="Original y = x^2")
    plt.title("Original Function (y = x^2)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    freq_range = [2,4,10]
    for freq in freq_range:
        # Define the sampled times and frequencies
        sampled_times = x_values
        frequencies = np.linspace(-freq/2.0, freq/2.0, 1000)

        # Apply FT to the sampled data
        ft_data = fourier_transform(y_values, frequencies, sampled_times)
        #  plot the FT data
        plt.figure(figsize=(12, 6))
        plt.plot(frequencies, np.sqrt(ft_data[0]**2 + ft_data[1]**2))
        plt.title("Frequency Spectrum of y = x^2")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.show()

        # Reconstruct the signal from the FT data
        reconstructed_y_values = inverse_fourier_transform(ft_data, frequencies, sampled_times)
        # Plot the original and reconstructed functions for comparison
        plt.figure(figsize=(12, 4))
        plt.plot(x_values, y_values, label="Original y = x^2", color="blue")
        plt.plot(sampled_times, reconstructed_y_values, label="Reconstructed y = x^2", color="red", linestyle="--")
        plt.title("Original vs Reconstructed Function (y = x^2)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
