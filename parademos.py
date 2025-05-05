# Sample from the trained model

n_images = 3

check_point = torch.load('check_point.pth', map_location=device)
score_model.load_state_dict(check_point)

def backward_drift_coefficient(x_t, t, diffusion_coefficient):
    g = diffusion_coefficient(t).view(-1, 1, 1, 1)
    return -g**2 * score_model(x_t, t)

diffusion_coefficient = diffusion_process.diffusion_coefficient

T = 1.0
image_T = torch.randn(n_images, 1, 28, 28).to(device)



with torch.no_grad():
    times, synthetic_images_t = predictor_corrector_integrator(
    image_T,
    t_0 = T,
    t_end = 1.0e-3,
    n_steps=500,
    drift_coefficient=partial(
        backward_drift_coefficient,
        diffusion_coefficient=diffusion_coefficient,
    ),
    diffusion_coefficient=diffusion_coefficient,
    score_function=score_model,  
    n_corrector_steps = 1,
    corrector_step_size = 0.01

)

print(type(synthetic_images_t))
print(synthetic_images_t.shape) 



from samplers import euler_maruyama_integrator
import torch
from noise_schedules import LinearSchedule
from diffusion import VPProcess

# Crear el noise schedule lineal
schedule = LinearSchedule(beta_min=0.1, beta_max=20.0)

# Instanciar el proceso de difusión Variance Preserving
vp_process = VPProcess(noise_schedule=schedule)

# Definir funciones de drift y diffusion desde la clase VPProcess
def drift(x_t, t):
    return vp_process.drift_coefficient(x_t, t)

def diffusion(t):
    return vp_process.diffusion_coefficient(t)

# Parámetros de integración
t_0 = 0.0
t_end = 1.0
n_steps = 501

# Integrar usando Euler-Maruyama
times, images_t = euler_maruyama_integrator(
    images_0,
    t_0,
    t_end,
    n_steps,
    drift_coefficient=drift,
    diffusion_coefficient=diffusion,
)

# Mostrar información de salida
print(type(images_0))
print(images_0.shape)

print(type(images_t))
print(images_t.shape)


# Train model

from torch.optim import Adam
import torchvision.transforms as transforms
from tqdm.notebook import trange

batch_size = 32

data_loader = DataLoader(
    data_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_threads,
)

#  [TO DO: Comment each line of code]

learning_rate = 1.0e-3
optimizer = Adam(score_model.parameters(), lr=learning_rate)

n_epochs =  10
tqdm_epoch = trange(n_epochs)

for epoch in tqdm_epoch:
    avg_loss = 0.0
    num_items = 0
    for x, y in data_loader:
        x = x.to(device)
        loss = diffusion_process.loss_function(score_model, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

    torch.save(score_model.state_dict(), 'check_point.pth')




# Train model

from torch.optim import Adam
import torchvision.transforms as transforms
from tqdm.notebook import trange

batch_size = 32

data_loader = DataLoader(
    data_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_threads,
)

#  [TO DO: Comment each line of code]

learning_rate = 1.0e-3
optimizer = Adam(score_model.parameters(), lr=learning_rate)

n_epochs =  10
tqdm_epoch = trange(n_epochs)

for epoch in tqdm_epoch:
    avg_loss = 0.0
    num_items = 0
    for x, y in data_loader:
        x = x.to(device)
        loss = diffusion_process.loss_function(score_model, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

    torch.save(score_model.state_dict(), 'check_point.pth')

    # Sample from the trained model


n_images = 3

check_point = torch.load('check_point.pth', map_location=device)
score_model.load_state_dict(check_point)

diffusion_coefficient = diffusion_process.diffusion_coefficient

T = 1.0 - 1e-3
image_T = torch.randn(n_images, 1, 28, 28).to(device)


with torch.no_grad():
    times, synthetic_images_t = predictor_corrector_integrator(
        image_T,
        t_0=T,
        t_end=1.0e-3,
        n_steps=2000,
        drift_coefficient=lambda x_t, t: diffusion_process.reverse_drift(x_t, t, score_model),
        diffusion_coefficient=diffusion_coefficient,
        score_function=score_model,  
        n_corrector_steps=10,
        corrector_step_size=0.0001
    )

print(type(synthetic_images_t))
print(synthetic_images_t.shape)

# Sample from the trained model

n_images = 3
target_class = torch.tensor([0] * n_images).to(device)  # Ejemplo: generar el dígito 7
score_function = lambda x_t, t: score_model(x_t, t, target_class)
drift_coefficient = lambda x_t, t: diffusion_process.reverse_drift(x_t, t, lambda x_t_, t_: score_model(x_t_, t_, target_class))

check_point = torch.load('check_point.pth', map_location=device)
score_model.load_state_dict(check_point)

diffusion_coefficient = diffusion_process.diffusion_coefficient

T = 1.0 - 1e-3
image_T = torch.randn(n_images, 1, 28, 28).to(device)


with torch.no_grad():
    times, synthetic_images_t = predictor_corrector_integrator(
        image_T,
        t_0=T,
        t_end=1.0e-3,
        n_steps=2000,
        drift_coefficient=drift_coefficient,
        diffusion_coefficient=diffusion_coefficient,
        score_function=score_function,
        n_corrector_steps=0,
        corrector_step_size=0.0001
    )


print(type(synthetic_images_t))
print(synthetic_images_t.shape)


n_images = 3

check_point = torch.load('check_point.pth', map_location=device)
score_model.load_state_dict(check_point)

diffusion_coefficient = diffusion_process.diffusion_coefficient

T = 1.0 - 1e-3
image_T = torch.randn(n_images, 1, 28, 28).to(device)


with torch.no_grad():
    times, synthetic_images_t = predictor_corrector_integrator(
        image_T,
        t_0=T,
        t_end=0- 1e-3,
        n_steps=2000,
        drift_coefficient=lambda x_t, t: diffusion_process.reverse_drift(x_t, t, score_model),
        diffusion_coefficient=diffusion_coefficient,
        score_function=score_model,
        n_corrector_steps=10,
        corrector_step_size=0.0001,
    )

print(type(synthetic_images_t))
print(synthetic_images_t.shape)


import matplotlib.pyplot as plt
# Crear instancia
schedule = CosineSchedule()

# Definir malla de tiempo (resolución reducida)
t_vals = torch.linspace(0.001, 1.0, 100)

# Varianza aproximada
approx_sigma = torch.sqrt(1 - schedule.alphas_cumprod(t_vals))

# Varianza exacta (con integración)
integrated_beta = schedule.integrated_beta(t_vals)
exact_sigma = torch.sqrt(1 - torch.exp(-integrated_beta))

# Graficar comparación
plt.figure(figsize=(8, 5))
plt.plot(t_vals.numpy(), approx_sigma.numpy(), label=r"$\sqrt{1 - \bar{\alpha}(t)}$ (aproximado)")
plt.plot(t_vals.numpy(), exact_sigma.numpy(), label=r"$\sqrt{1 - e^{-\int_0^t \beta(s)ds}}$ (exacto)", linestyle="--")
plt.title("Comparación de varianza: aproximación vs. integral exacta")
plt.xlabel("t")
plt.ylabel(r"$\sigma(t)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



n_images = 3

mask = torch.randint(0, 2, (n_images, 1, 28, 28), device=device, dtype=torch.float32)
check_point = torch.load('check_point.pth', map_location=device)
score_model.load_state_dict(check_point)

diffusion_coefficient = diffusion_process.diffusion_coefficient

T = 1.0 - 1e-3
image_T = torch.zeros(n_images, 1, 28, 28).to(device)
T_tensor = torch.tensor([T], device=device)

with torch.no_grad():
    times, synthetic_images_t = euler_maruyama_integrator(
        image_T,
        t_0=T,
        t_end= 0,
        n_steps= 5000,
        drift_coefficient=lambda x_t, t: diffusion_process.reverse_drift(x_t, t, score_model),
        diffusion_coefficient=diffusion_coefficient,
        prior = (lambda x_t, T_tensor: diffusion_process.mu_t(x_t, T_tensor), 
                 lambda T_tensor: diffusion_process.sigma_t(T_tensor))
   )

print(type(synthetic_images_t))
print(synthetic_images_t.shape)