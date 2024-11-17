# train_ai_gfx
import pygame
import time
from collections import deque
from stable_baselines3 import PPO
from train_ai_data import TemperatureControlEnv  # Varmista, että polku on oikein

# Pygame alustukset
pygame.init()

# Ikkunan asetukset
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (30, 30, 30)
TEMP_BAR_COLOR = (100, 200, 100)
SETPOINT_COLOR = (200, 100, 100)
VALVE_COLOR = (100, 100, 200)
HISTOGRAM_COLOR = (100, 150, 250)

# Ikkunan luonti
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Temperature Control AI")

# Fontti tekstien näyttämiseen
font = pygame.font.Font(None, 36)

# Histogrammin asetukset
HISTOGRAM_WIDTH = 300
HISTOGRAM_HEIGHT = 200
HISTOGRAM_MAX_POINTS = 50  # Montako pistettä näytetään kerralla
HISTOGRAM_X = WIDTH - HISTOGRAM_WIDTH - 10
HISTOGRAM_Y = 10

# Histogrammidata
temperature_history = deque(maxlen=HISTOGRAM_MAX_POINTS)

def render_environment(screen, current_temp, setpoint_temp, valve_adjust):
    # Tyhjennä ruutu
    screen.fill(BACKGROUND_COLOR)
    
    # Lämpötilapalkin piirtäminen
    temp_bar_height = int((current_temp + 50) / 150 * HEIGHT)  # Skaalataan lämpötila palkkiin
    pygame.draw.rect(screen, TEMP_BAR_COLOR, (WIDTH // 4, HEIGHT - temp_bar_height, 100, temp_bar_height))
    
    # Tavoitelämpötilan viiva
    setpoint_y = HEIGHT - int((setpoint_temp + 50) / 150 * HEIGHT)
    pygame.draw.line(screen, SETPOINT_COLOR, (WIDTH // 4 - 20, setpoint_y), (WIDTH // 4 + 120, setpoint_y), 3)
    
    # Venttiilin säätötekstin näyttö
    valve_text = font.render(f"Valve Adjust: {valve_adjust:.2f}%", True, VALVE_COLOR)
    screen.blit(valve_text, (WIDTH // 2, HEIGHT - 50))
    
    # Näytä lämpötila ja tavoitelämpötila
    temp_text = font.render(f"Temp: {current_temp:.2f}°C | Setpoint: {setpoint_temp:.2f}°C", True, (255, 255, 255))
    screen.blit(temp_text, (50, 50))
    
    # Päivitä näyttö
    pygame.display.flip()

def render_histogram(screen, temperature_history, setpoint_temp):
    # Histogrammin taustan piirtäminen
    pygame.draw.rect(screen, (50, 50, 50), (HISTOGRAM_X, HISTOGRAM_Y, HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT))
    
    # Skaalataan lämpötilapisteet histogrammiin
    max_temp = 100
    min_temp = 0
    scaled_points = [
        ((HISTOGRAM_X + i * (HISTOGRAM_WIDTH // HISTOGRAM_MAX_POINTS)),
         HISTOGRAM_Y + HISTOGRAM_HEIGHT - int((temp - min_temp) / (max_temp - min_temp) * HISTOGRAM_HEIGHT))
        for i, temp in enumerate(temperature_history)
    ]
    
    # Piirrä historialliset lämpötilat viivana
    if len(scaled_points) > 1:
        pygame.draw.lines(screen, HISTOGRAM_COLOR, False, scaled_points, 2)
    
    # Tavoitelämpötilan viivan piirto histogrammiin
    setpoint_y = HISTOGRAM_Y + HISTOGRAM_HEIGHT - int((setpoint_temp - min_temp) / (max_temp - min_temp) * HISTOGRAM_HEIGHT)
    pygame.draw.line(screen, SETPOINT_COLOR, (HISTOGRAM_X, setpoint_y), (HISTOGRAM_X + HISTOGRAM_WIDTH, setpoint_y), 2)

    # Päivitä näyttö
    pygame.display.flip()

def main():
    # Ladataan koulutettu PPO-malli
    model = PPO.load("ppo_temperature_control")
    
    # Luodaan ympäristö
    env = TemperatureControlEnv()
    
    # Aloita Pygame-silmukka
    running = True
    episode = 1  # Episodien laskuri
    try:
        clock = pygame.time.Clock()
        fps = 10  # Päivitystiheys (FPS)
        while running:
            obs, _ = env.reset()
            done = False
            total_reward = 0
            active_after_target = 0  # Kohinan hallinnan laskuri

            # Tyhjennä histogrammi jokaisen episodin alussa
            temperature_history.clear()

            print(f"Episodi {episode} alkaa...")
            
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        done = True
                
                # Ennusta toiminto AI:n avulla
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Näytä nykyinen ympäristö
                current_temp, setpoint_temp = obs
                valve_adjust = info["valve_adjust"]
                render_environment(screen, current_temp, setpoint_temp, valve_adjust)
                
                # Päivitä histogrammi ja piirrä
                temperature_history.append(current_temp)
                render_histogram(screen, temperature_history, setpoint_temp)
                
                total_reward += reward
                
                # Tarkista, jatkuuko episodi
                if terminated:
                    if active_after_target < 500:
                        terminated = False  # Jatka episodia
                    else:
                        done = True  # Episodi päättyy
                elif True:  # Tavoitelämpötilan tarkistus ohitetaan
                    active_after_target += 1
                    print(f"active_after_target: {active_after_target}")  # DEBUG: Seuraa laskuria
                    if active_after_target >= 100:  # Esim. 500 sykliä (50 sekuntia, jos fps = 10)
                        print("Aika täyttyi, siirrytään seuraavaan episodiin.")
                        done = True  # Päätetään episodi
                else:
                    if active_after_target > 0:
                        print("active_after_target nollattu.")
                    active_after_target = 0
                
                # Pieni viive
                time.sleep(0.1)
            print(f"Episodi {episode} päättyi, kokonaisscore: {total_reward:.2f}")
            terminated = False
            episode += 1
            
            # Lyhyt tauko episodien välillä
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nOhjelma keskeytetty käyttäjän toimesta (Ctrl+C).")
    
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()