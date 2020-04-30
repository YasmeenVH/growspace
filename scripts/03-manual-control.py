import cv2

from growspace.envs import GrowSpaceEnv

gse = GrowSpaceEnv()


def key2action(key):
    if key == ord('a'):
        return 0  # move left
    elif key == ord('d'):
        return 1  # move right
    elif key == ord('s'):
        return 2  # stay in place
    else:
        return None


while True:
    gse.reset()
    img = gse.get_observation(debug_show_scatter=True)
    cv2.imshow("plant", img)

    for _ in range(10):
        action = key2action(cv2.waitKey(-1))
        if action is None:
            quit()

        gse.step(action)
        cv2.imshow("plant", gse.get_observation(debug_show_scatter=True))