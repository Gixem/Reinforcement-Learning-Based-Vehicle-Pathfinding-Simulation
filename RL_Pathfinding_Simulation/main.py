import numpy as np
import random
import pygame
import time
import matplotlib.pyplot as plt
import pandas as pd

def save_q_table(self, filename="q_table.csv"):
    q_table_reshaped = self.q_table.reshape(self.GRID_SIZE * self.GRID_SIZE, -1)
    df = pd.DataFrame(q_table_reshaped, columns=["Up", "Down", "Left", "Right"])
    df.to_csv(filename, index=False)
    print(f"Q-Table saved to {filename}")

class QLearningAgent:
    def __init__(self, grid_size, window_size, epsilon, alpha, gamma, episodes):
        self.GRID_SIZE = grid_size
        self.WINDOW_SIZE = window_size
        self.CELL_SIZE = window_size // grid_size
        self.EPSILON = epsilon
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.EPISODES = episodes
        self.REWARD_GOAL = 10
        self.REWARD_MOVE = -1
        self.REWARD_INVALID = -10
        self.EPSILON_DECAY = 0.995
        self.EPSILON_MIN = 0.01
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.q_table = np.zeros((grid_size, grid_size, 4))
        self.agent_pos = [0, 0]
        self.GOAL = (grid_size - 1, grid_size - 1)
        self.OBSTACLES = [(3, 3, "barrier"), (5, 5, "cone"), (7, 7, "barrier")]
        self.total_rewards = []

        # Resimleri yükle
        self.road_image = pygame.image.load("road.png")
        self.road_image = pygame.transform.scale(self.road_image, (self.CELL_SIZE, self.CELL_SIZE))
        self.barrier_image = pygame.image.load("barrier.png")
        self.barrier_image = pygame.transform.scale(self.barrier_image, (self.CELL_SIZE, self.CELL_SIZE))
        self.cone_image = pygame.image.load("cone.png")
        self.cone_image = pygame.transform.scale(self.cone_image, (self.CELL_SIZE, self.CELL_SIZE))
        self.finish_image = pygame.image.load("finish.png")
        self.finish_image = pygame.transform.scale(self.finish_image, (self.CELL_SIZE, self.CELL_SIZE))
        self.car_image = pygame.image.load("car.png")
        self.car_image = pygame.transform.scale(self.car_image, (self.CELL_SIZE, self.CELL_SIZE))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.EPSILON:
            return random.choice(range(4))
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def update_q_table(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state[0], next_state[1]])
        current_q = self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] = current_q + self.ALPHA * (reward + self.GAMMA * max_future_q - current_q)

    def run_episode(self, screen):
        self.agent_pos = [0, 0]
        total_reward = 0
        steps = 0
        done = False

        MAX_STEPS = 500  # Maksimum adım sayısı

        while not done and steps < MAX_STEPS:
            # Kullanıcı olaylarını kontrol et
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            state = self.agent_pos.copy()
            action = self.choose_action(state)
            next_pos = [state[0] + self.actions[action][0], state[1] + self.actions[action][1]]

            # Hareket kontrolü
            if (
                0 <= next_pos[0] < self.GRID_SIZE
                and 0 <= next_pos[1] < self.GRID_SIZE
                and tuple(next_pos[:2]) not in [(o[0], o[1]) for o in self.OBSTACLES]
            ):
                self.agent_pos = next_pos
                reward = self.REWARD_MOVE
            else:
                reward = self.REWARD_INVALID

            if tuple(self.agent_pos) == self.GOAL:
                reward = self.REWARD_GOAL
                done = True

            self.update_q_table(state, action, reward, self.agent_pos)
            total_reward += reward
            steps += 1

            # Ekranı düzenli olarak güncelle
            self.draw_grid(screen)
            pygame.display.flip()

            # FPS kontrolü ile ekranı yavaşlat
            clock.tick(30)  # 30 FPS hızında çalışır

        self.EPSILON = max(self.EPSILON * self.EPSILON_DECAY, self.EPSILON_MIN)
        self.total_rewards.append(total_reward)
        return total_reward, steps


    def draw_grid(self, screen):
        screen.fill((255, 255, 255))

        # Zemin (yol) çizimi
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                screen.blit(self.road_image, (y * self.CELL_SIZE, x * self.CELL_SIZE))

        # Engelleri çiz
        for obstacle in self.OBSTACLES:
            obstacle_image = self.barrier_image if obstacle[2] == "barrier" else self.cone_image
            screen.blit(
                obstacle_image,
                (obstacle[1] * self.CELL_SIZE, obstacle[0] * self.CELL_SIZE),
            )

        # Hedefi çiz
        screen.blit(
            self.finish_image,
            (self.GOAL[1] * self.CELL_SIZE, self.GOAL[0] * self.CELL_SIZE),
        )

        # Ajanı (araba) çiz
        screen.blit(
            self.car_image,
            (self.agent_pos[1] * self.CELL_SIZE, self.agent_pos[0] * self.CELL_SIZE),
        )

    def plot_results(self):
        # Toplam ödülleri görselleştir
        plt.figure(figsize=(10, 5))
        plt.plot(self.total_rewards)
        plt.title("Total Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid()
        plt.show()

        # Q-Table Heatmap'i görselleştir
        plt.figure(figsize=(10, 5))
        q_table_sum = np.sum(self.q_table, axis=2)
        plt.imshow(q_table_sum, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label="Sum of Q-Values")
        plt.title("Q-Table Heatmap")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

        # Q-Table'ı Excel'e aktarma
        q_table_reshaped = self.q_table.reshape(self.GRID_SIZE * self.GRID_SIZE, -1)  # Q-Tabloyu 2D hale getir
        df = pd.DataFrame(q_table_reshaped, columns=["Up", "Down", "Left", "Right"])  # Sütun isimlerini ekle
        df.index.name = "State"  # Satır isimlerini 'State' olarak belirle
        df.to_csv("q_table.csv", index=True, float_format="%.4f")  # CSV olarak kaydet
        print("Q-Table saved to q_table.csv")


if __name__ == "__main__":
    pygame.init()
    grid_size = 10
    window_size = 600
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Q-Learning Pathfinding Simulation")
    clock = pygame.time.Clock()

    agent = QLearningAgent(grid_size=grid_size, window_size=window_size, epsilon=1.0, alpha=0.1, gamma=0.9, episodes=300)

    for episode in range(agent.EPISODES):
        total_reward, steps = agent.run_episode(screen)
        print(f"Episode {episode + 1}/{agent.EPISODES} - Total Reward: {total_reward}, Steps: {steps}")

    pygame.quit()
    # Eğitim sonuçlarını görselleştir
    agent.plot_results()
    # Q-Table'ı Excel'e kaydet
    agent.save_q_table()
