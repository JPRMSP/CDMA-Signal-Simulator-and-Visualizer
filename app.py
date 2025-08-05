import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# BACKEND FUNCTIONS
# ----------------------------

def generate_walsh_codes(n):
    """Generate Walsh codes of order n (must be power of 2)"""
    if n == 1:
        return np.array([[1]])
    else:
        h = generate_walsh_codes(n // 2)
        top = np.hstack((h, h))
        bottom = np.hstack((h, -h))
        return np.vstack((top, bottom))

def generate_pn_sequence(length, seed=1):
    """Generate a simple PN sequence using a random seed"""
    np.random.seed(seed)
    return np.random.choice([-1, 1], size=length)

def spread_signal(data_bits, spreading_code):
    """Spread data using the given spreading code"""
    return np.repeat(data_bits, len(spreading_code)) * np.tile(spreading_code, len(data_bits))

def add_awgn_noise(signal, snr_db):
    """Add Gaussian noise based on SNR (in dB)"""
    snr_linear = 10 ** (snr_db / 10)
    power = np.mean(signal ** 2)
    noise_power = power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def despread_signal(received_signal, spreading_code):
    """Despread the received signal and make bit decisions"""
    despread = np.sum(received_signal.reshape(-1, len(spreading_code)) * spreading_code, axis=1)
    return np.where(despread >= 0, 1, -1)

# ----------------------------
# STREAMLIT FRONTEND
# ----------------------------

st.set_page_config(page_title="CDMA Signal Simulator", layout="centered")
st.title("📡 CDMA Signal Simulator and Visualizer")

st.markdown("""
This simulator demonstrates key CDMA concepts:
- Walsh and PN spreading codes
- Multi-user transmission
- AWGN channel noise
- Despreading and signal recovery

Made for **FI1924 - CDMA Signal Detection** (Anna University, Regulation 2021).
---
""")

# Sidebar controls
st.sidebar.title("🔧 Simulation Settings")
num_users = st.sidebar.slider("Number of Users", 1, 4, 2)
code_type = st.sidebar.selectbox("Spreading Code Type", ["Walsh Code", "PN Code"])
snr_db = st.sidebar.slider("SNR (dB)", 0, 30, 10)
data_length = st.sidebar.slider("Data Bits per User", 1, 8, 4)

# Generate random data bits for each user
st.subheader("🔣 Random Data Bits for Each User")
user_data = []
for user in range(num_users):
    bits = np.random.choice([-1, 1], size=data_length)
    user_data.append(bits)
    st.write(f"User {user+1}: {bits}")

# Spread signals
spreaded_signals = []
codes = []
code_len = 2 ** int(np.ceil(np.log2(data_length)))  # for Walsh, next power of 2

st.subheader("🔍 Spreading Codes and Spread Signals")

for user in range(num_users):
    st.markdown(f"**User {user+1}**")
    
    if code_type == "Walsh Code":
        walsh_codes = generate_walsh_codes(code_len)
        code = walsh_codes[user]
    else:
        code = generate_pn_sequence(data_length * 8, seed=user)
    
    codes.append(code)
    spreaded = spread_signal(user_data[user], code)
    spreaded_signals.append(spreaded)

    fig, ax = plt.subplots(figsize=(6, 1))
    ax.plot(code, drawstyle='steps-pre')
    ax.set_title(f"Spreading Code (User {user+1})")
    ax.set_yticks([-1, 1])
    st.pyplot(fig)

# Combine signals and simulate AWGN channel
combined_signal = np.sum(spreaded_signals, axis=0)
noisy_signal = add_awgn_noise(combined_signal, snr_db)

st.subheader("📡 Combined Signal with Noise (Channel Output)")
fig, ax = plt.subplots()
ax.plot(noisy_signal, label="Received Signal")
ax.set_title("Received CDMA Signal (Noisy)")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")
st.pyplot(fig)

# Despread signal for User 1
st.subheader("🧩 Recovered Signal for User 1")
recovered_bits = despread_signal(noisy_signal, codes[0])
st.write("Original Bits:", user_data[0])
st.write("Recovered Bits:", recovered_bits)

accuracy = np.mean(recovered_bits == user_data[0]) * 100
st.success(f"✅ Bit Recovery Accuracy for User 1: {accuracy:.2f}%")

st.markdown("---")
st.markdown("Made with ❤️ by ECE Student | Project aligned with **CDMA Signal Detection (FI1924)**")
