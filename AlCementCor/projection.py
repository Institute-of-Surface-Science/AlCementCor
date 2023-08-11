def project_onto_plane(A, B, C, displacement):
    AB = B - A
    AC = C - A
    normal = np.cross(AB, AC)
    normal_magnitude = np.linalg.norm(normal)
    if normal_magnitude == 0:
        print("Points A, B, and C are collinear; they do not define a plane")
        # Handle this special case as needed for your application
        return displacement  # This is just an example; you might want to do something else
    normal_hat = normal / normal_magnitude
    return displacement - (np.dot(displacement, normal_hat) * normal_hat)