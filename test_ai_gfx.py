# train_ai_gfx
# Licensed under the MIT License.
# Copyright (c) 2024 Timo Heimonen.
# See the LICENSE file in the project root for more details.

import pygame
import time
from collections import deque
from stable_baselines3 import PPO
from train_ai_data import TemperatureControlEnv  # Ensure the path is correct

# Pygame initialization
pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (30, 30, 30)
TEMP_BAR_COLOR = (100, 200, 100)
SETPOINT_COLOR = (200, 100, 100)
VALVE_COLOR = (100, 100, 200)
HISTOGRAM_COLOR = (100, 150, 250)

# Create the window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Temperature Control AI")

# Font for displaying text
font = pygame.font.Font(None, 36)

# Histogram settings
HISTOGRAM_WIDTH = 300
HISTOGRAM_HEIGHT = 200
HISTOGRAM_MAX_POINTS = 50  # Number of points to display
HISTOGRAM_X = WIDTH - HISTOGRAM_WIDTH - 10
HISTOGRAM_Y = 10

# Histogram data
temperature_history = deque(maxlen=HISTOGRAM_MAX_POINTS)

def render_environment(screen, current_temp, setpoint_temp, valve_adjust):
    # Clear the screen
    screen.fill(BACKGROUND_COLOR)
    
    # Draw the temperature bar
    temp_bar_height = int((current_temp + 50) / 150 * HEIGHT)  # Scale temperature to the bar
    pygame.draw.rect(screen, TEMP_BAR_COLOR, (WIDTH // 4, HEIGHT - temp_bar_height, 100, temp_bar_height))
    
    # Draw the setpoint line
    setpoint_y = HEIGHT - int((setpoint_temp + 50) / 150 * HEIGHT)
    pygame.draw.line(screen, SETPOINT_COLOR, (WIDTH // 4 - 20, setpoint_y), (WIDTH // 4 + 120, setpoint_y), 3)
    
    # Display valve adjustment text
    valve_text = font.render(f"Valve Adjust: {valve_adjust:.2f}%", True, VALVE_COLOR)
    screen.blit(valve_text, (WIDTH // 2, HEIGHT - 50))
    
    # Display temperature and setpoint
    temp_text = font.render(f"Temp: {current_temp:.2f}°C | Setpoint: {setpoint_temp:.2f}°C", True, (255, 255, 255))
    screen.blit(temp_text, (50, 50))
    
    # Update the display
    pygame.display.flip()

def render_histogram(screen, temperature_history, setpoint_temp):
    # Draw histogram background
    pygame.draw.rect(screen, (50, 50, 50), (HISTOGRAM_X, HISTOGRAM_Y, HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT))
    
    # Scale temperature points to the histogram
    max_temp = 100
    min_temp = 0
    scaled_points = [
        ((HISTOGRAM_X + i * (HISTOGRAM_WIDTH // HISTOGRAM_MAX_POINTS)),
         HISTOGRAM_Y + HISTOGRAM_HEIGHT - int((temp - min_temp) / (max_temp - min_temp) * HISTOGRAM_HEIGHT))
        for i, temp in enumerate(temperature_history)
    ]
    
    # Draw the historical temperature as a line
    if len(scaled_points) > 1:
        pygame.draw.lines(screen, HISTOGRAM_COLOR, False, scaled_points, 2)
    
    # Draw the setpoint line on the histogram
    setpoint_y = HISTOGRAM_Y + HISTOGRAM_HEIGHT - int((setpoint_temp - min_temp) / (max_temp - min_temp) * HISTOGRAM_HEIGHT)
    pygame.draw.line(screen, SETPOINT_COLOR, (HISTOGRAM_X, setpoint_y), (HISTOGRAM_X + HISTOGRAM_WIDTH, setpoint_y), 2)

    # Update the display
    pygame.display.flip()

def main():
    # Load the trained PPO model
    model = PPO.load("ppo_temperature_control")
    
    # Create the environment
    env = TemperatureControlEnv()
    
    # Start the Pygame loop
    running = True
    episode = 1  # Episode counter
    try:
        clock = pygame.time.Clock()
        fps = 10  # Refresh rate (FPS)
        while running:
            obs, _ = env.reset()
            done = False
            total_reward = 0
            active_after_target = 0  # Counter for noise handling

            # Clear the histogram at the beginning of each episode
            temperature_history.clear()

            print(f"Episode {episode} begins...")
            
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        done = True
                
                # Predict the action using AI
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Display the current environment
                current_temp, setpoint_temp = obs
                valve_adjust = info["valve_adjust"]
                render_environment(screen, current_temp, setpoint_temp, valve_adjust)
                
                # Update and draw the histogram
                temperature_history.append(current_temp)
                render_histogram(screen, temperature_history, setpoint_temp)
                
                total_reward += reward
                
                # Check if the episode continues
                if terminated:
                    if active_after_target < 500:
                        terminated = False  # Continue the episode
                    else:
                        done = True  # Episode ends
                elif True:  # Skip setpoint temperature check
                    active_after_target += 1
                    print(f"active_after_target: {active_after_target}")  # DEBUG: Monitor counter
                    if active_after_target >= 100:  # Example: 500 cycles (50 seconds if fps = 10)
                        print("Time limit reached, moving to the next episode.")
                        done = True  # End the episode
                else:
                    if active_after_target > 0:
                        print("active_after_target reset.")
                    active_after_target = 0
                
                # Small delay
                time.sleep(0.1)
            print(f"Episode {episode} ended, total score: {total_reward:.2f}")
            terminated = False
            episode += 1
            
            # Short pause between episodes
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by the user (Ctrl+C).")
    
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()