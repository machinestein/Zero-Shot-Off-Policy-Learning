import numpy as np

def get_donut_mask(X, Y, inner_r=0.25, outer_r=1.5):
    R = np.sqrt(X**2 + Y**2)
    return (R >= inner_r) & (R <= outer_r)

def two_circles_reward(X, Y, r1=0.3, r2=1.0, sigma=0.4):
    R = np.sqrt(X**2 + Y**2)
    dist_to_c1 = np.abs(R - r1)
    dist_to_c2 = np.abs(R - r2)
    min_dist = np.minimum(dist_to_c1, dist_to_c2)
    Z = 1.0 - 2.0 * (min_dist / sigma)
    Z = np.maximum(Z, -1.0)
    mask = get_donut_mask(X, Y)
    return np.where(mask, Z, -1.0)

def square_reward(X, Y, side=1.0, sigma=0.4):
    s = side / 2.0
    abs_x, abs_y = np.abs(X), np.abs(Y)
    dx = np.maximum(0, abs_x - s)
    dy = np.maximum(0, abs_y - s)
    dist_outside = np.sqrt(dx**2 + dy**2)
    dist_inside = np.minimum(s - abs_x, s - abs_y)
    is_outside_square = (abs_x > s) | (abs_y > s)
    dist = np.where(is_outside_square, dist_outside, np.abs(dist_inside))
    Z = 1.0 - 2.0 * (dist / sigma)
    Z = np.maximum(Z, -1.0)
    mask = get_donut_mask(X, Y)
    return np.where(mask, Z, -1.0)

def cross_reward(X, Y, sigma=0.3):
    dist = np.minimum(np.abs(X), np.abs(Y))
    Z = 1.0 - 2.0 * (dist / sigma)
    Z = np.maximum(Z, -1.0)
    mask = get_donut_mask(X, Y)
    return np.where(mask, Z, -1.0)

def grid_reward(X, Y, spacing=0.5, sigma=0.2):
    dist_x = np.abs((X + spacing/2) % spacing - spacing/2)
    dist_y = np.abs((Y + spacing/2) % spacing - spacing/2)
    dist = np.sqrt(dist_x**2 + dist_y**2)
    Z = 1.0 - 2.0 * (dist / sigma)
    Z = np.maximum(Z, -1.0)
    mask = get_donut_mask(X, Y)
    return np.where(mask, Z, -1.0)
