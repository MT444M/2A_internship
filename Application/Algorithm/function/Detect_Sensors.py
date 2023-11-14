import numpy as np
from sklearn.cluster import DBSCAN
import itertools
from typing import List, Tuple
import cv2





def create_partial_grid(num_lines, num_corners_per_line, width, distance, x_0, y_0):
    """
    Create a partial grid of corners based on the specified parameters.

    Args:
        num_lines (int): Number of lines in the grid.
        num_corners_per_line (int): Number of corners per line.
        width (int): Width of the sensor.
        distance (int): Distance between sensors.
        x_0 (int): Initial x-coordinate of the first corner.
        y_0 (int): Initial y-coordinate of the first corner.

    Returns:
        list: A 2D list representing the partial grid of corners.
    """

    grid = []
    previous_line = create_line(num_corners_per_line, width, distance, x_0, y_0)
    grid.append(previous_line)

    # Create the remaining lines
    for line in range(1, num_lines):
        new_line = []
        for corner in range(num_corners_per_line):
            # Determine the coordinates of the new corner
            if line % 2 == 1:
                # If line number is odd, increment x-coordinate by width
                new_corner = (previous_line[corner][0] + width, previous_line[corner][1])
            else:
                # If line number is even, increment x-coordinate by distance
                new_corner = (previous_line[corner][0] + distance, previous_line[corner][1])
            new_line.append(new_corner)

        # Add the new line to the grid
        grid.append(new_line)
        previous_line = new_line

    return grid



def create_line(num_corners, width, distance, x_0, y_0):
    """
    Create a line of corners based on the specified parameters.

    Args:
        num_corners (int): Number of corners in the line.
        width (int): Width of the sensor.
        distance (int): Distance between sensors.
        x_0 (int): Initial x-coordinate of the first corner.
        y_0 (int): Initial y-coordinate of the first corner.

    Returns:
        list: A list representing the line of corners.
    """

    corners = []
    x = x_0
    y = y_0

    corners.append((x, y))  # Adding the first corner

    # Iterate over the remaining corners
    for corner in range(1, num_corners):
        if corner % 2 == 1:
            # If corner number is odd, increment y-coordinate by width
            x = corners[corner-1][0]
            y = corners[corner-1][1] + width
        else:
            # If corner number is even, increment y-coordinate by distance
            x = corners[corner-1][0]
            y = corners[corner-1][1] + distance

        corners.append((x, y))

    return corners

def check_detect_criteria(corners_matrix, nbre_corners, nbre_lines):
    """
    Check if the criteria are satisfied for the sensor points.
    - There should be at least 18 lines of points.
    - Each line should have at least 16 points.
    - The number of lines and the number of points per line should be even.
    - param: nbre_corners

    Args:
        corners_matrix (list): Matrix of sensor points.
        nbre_corners : the minimum nombre of sensors of the y-axis (/2)
        nbre_lines : the minimum nombre of sensors of the x-axis (/2)
    Returns:
        bool: True if the criteria are satisfied, False otherwise.
    """
    matrix_len = len(corners_matrix)
    if matrix_len < nbre_lines or matrix_len % 2 != 0:
        return False

    for line_points in corners_matrix:
        line_len = len(line_points)
        if line_len < nbre_corners or line_len % 2 != 0:
            return False

    return True





def get_Coords_sensors(img: np.ndarray, params_Canny: Tuple[int, int], nbre_lines_selected: int):
    """
    Detect the coordinates of the corners of the sensor matrix from a processed image.

    Args:
        img (np.ndarray): The processed image.
        params_Canny (Tuple[int, int]): Canny filter parameters (low threshold, high threshold).
        nbre_lines_selected (int): Number of selected lines for the sensor matrix detection.

    Returns:
        List[Tuple[int, int]]: The coordinates of the corners of the sensor matrix.
    """

    # Detect edges using the Canny algorithm
    edges = cv2.Canny(img, params_Canny[0], params_Canny[1])

    # Apply Hough transform on the detected edges
    lines = cv2.HoughLines(edges, 1, np.pi / 720.0, 50, np.array([]), 0, 0)

    # Select only nbre_lines_selected lines from the detected lines
    lines_best = np.squeeze(lines)[:nbre_lines_selected]

    # Calculate the coordinates of the intersections between the lines
    intersections, intersections_info, parallel_sets_list = get_all_line_intersection(lines_best, img.shape)

    # Get the line clusters, i.e., sets of lines involved in the same intersection
    intersecting_clusters = get_intersecting_line_clusters(intersections, intersections_info)

    # Get the indices of parallel line pairs grouped into clusters
    parallel_clusters = get_parallel_line_clusters(lines_best, parallel_sets_list)

    # Return the two main sets of parallel lines
    # Corresponding to the vanishing lines of the sensor matrix
    best_cluster_pair = select_best_performing_cluster_pair(lines_best, intersecting_clusters, parallel_clusters)

    # Get the lines corresponding to the vanishing lines
    cluster_means = [cluster_mean_hessfixed(lines_best, best_cluster_pair[0]),
                     cluster_mean_hessfixed(lines_best, best_cluster_pair[1])]

    # Eliminate line clusters
    best_cluster_pair_duplicate_eliminated = [
        cluster_eliminate_duplicate(lines_best, best_cluster_pair[0], cluster_means[1], img.shape),
        cluster_eliminate_duplicate(lines_best, best_cluster_pair[1], cluster_means[0], img.shape)]

    # Calculate the corners of each box
    all_corners_in_Matrix = get_intersections_between_clusters(best_cluster_pair_duplicate_eliminated[0],
                                                               best_cluster_pair_duplicate_eliminated[1],
                                                               img.shape)

    return all_corners_in_Matrix


def processing_corners(all_corners_in_matrix):
    """
    Complete the missing corners for each line of corners and return a sorted matrix by x, then by y.

    Args:
        all_corners_in_matrix (list): A list of corners extracted from an image.

    Returns:
        numpy.ndarray: A completed matrix of corners for each line, sorted by x, then by y.
    """

    # Sort the points of each line by x
    sorted_list = sorted(all_corners_in_matrix, key=lambda line: np.mean(line, axis=0)[0])
    # Remove lines containing negative values
    sorted_list = [line for line in sorted_list if all(x >= 0 for sublist in line for x in sublist)]

    # Sort the points of each line by y
    sorted_list = [np.array(line)[np.argsort(np.array(line)[:, 1])] for line in sorted_list]

    # Complete the missing lines
    lines_completed = []
    for line in sorted_list:
        if line.shape[0] == 15:
            # Calculate the horizontal distance between the first two points
            dist = line[1][1] - line[0][1]

            # Add the missing sixth point at the sixth position of the line
            x = line[0][0]
            y = line[4][1] + dist
            new_point = np.array([[x, y]], dtype=np.int32)
            line = np.insert(line, 5, new_point, axis=0)

        lines_completed.append(line)

    return np.array(lines_completed)


def extract_boxes_from_lines(lines):
    """
    Extract the boxes from the lines by calculating their adjacent intersections.

    Args:
        lines (list): A list of lines represented by a list of coordinate points.

    Returns:
        list: A list of boxes represented by a list of their four corners.

    Raises:
        ValueError: If the number of lines is odd or the number of points in the two adjacent lines is different.
    """

    # Check if the number of lines is even
    if len(lines) % 2 != 0:
        raise ValueError("The number of lines is not even.")

    boxes = []
    for i in range(0, len(lines), 2):
        line1 = lines[i]
        line2 = lines[i + 1]
        if len(line1) != len(line2):
            raise ValueError("The number of points in the two lines is different.")

    for i in range(0, len(lines), 2):
        line1 = lines[i]
        line2 = lines[i + 1]
        if len(line1) != len(line2):
            raise ValueError("The number of points in the two lines is different.")

        for j in range(0, len(line1) - 1, 2):
            # The corners of box i
            x1 = line1[j][0]
            y1 = line1[j][1]
            x2 = line1[j + 1][0]
            y2 = line1[j + 1][1]
            x3 = line2[j + 1][0]
            y3 = line2[j + 1][1]
            x4 = line2[j][0]
            y4 = line2[j][1]
            boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    return boxes


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def intersection(line1, line2, img_shape):
    """
        Calcule le point d'intersection entre deux lignes sur une image.

        Args:
            line1 (tuple): Coordonnées polaires de la première ligne (rho, theta).
            line2 (tuple): Coordonnées polaires de la deuxième ligne (rho, theta).
            img_shape (tuple): Dimensions de l'image (hauteur, largeur).

        Returns:
            list: Coordonnées (x, y) du point d'intersection entre les deux lignes, ou (-1, -1) si la résolution échoue ou si le point d'intersection est en dehors de l'image.
        """

    # On récupère les coordonnées polaires des lignes 1 et 2 :
    rho1, theta1 = line1
    rho2, theta2 = line2

    # On construit la matrice A et le vecteur b relative au système d'équation
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    try:
        # On résout le système d'équation linéaire Ax+b pour trouver le point d'intersection
        x0, y0 = np.linalg.solve(A, b)
    except:
        # Dans le cas où la résolution échoue on renvoie les coordonnées (-1,-1)
        x0, y0 = (-1, -1)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    # On vérifie si les intersections sont en dehors des limites de l'image dans quel cas on renvoie (-1,1)
    if abs(y0) > img_shape[0] * parallel_threshold or abs(x0) > img_shape[0] * parallel_threshold:
        x0, y0 = (-1, -1)
    return [x0, y0]


def split_clusters_using_labels(all_clusters, labels):
    """
       Sépare les clusters de droites détectés en groupes de droites associées.

       Parameters:
       -----------
           all_clusters: list
               Liste contenant tous les clusters de droites détectés.
           labels: numpy.ndarray
               Tableau des labels associés à chaque droite.

       Returns:
       --------
           cluster_list: list
               Liste de groupes de droites associées pour chaque cluster.

       """
    cluster_list = []
    # On crée pour chaque cluster, on crée un ensemble de toutes les droites impliquées dans le cluster
    for cluster_id in range(max(labels) + 1):
        mask = (labels == cluster_id)
        # On cree un ens des paires de droite impliquée dans le cluster
        cluster_list.append(np.array(all_clusters)[mask])

    return cluster_list


def fix_hessian_form(line, reverse=False):
    """
        Convertit une ligne de forme (rho, alpha) en une forme normalisée,
        où rho est positif et alpha est compris entre -pi/2 et pi/2.

        Args:
            line (tuple): Tuple de forme (rho, alpha).
            reverse (bool): Indique si l'on doit inverser le signe de rho et alpha.

        Returns :
            tuple : Tuple de forme normalisée (rho, alpha).
        """
    if not reverse and line[0] < 0:
        new_rho = - (line[0])
        new_alpha = -(np.pi - line[1])
        return new_rho, new_alpha
    elif reverse and line[1] < 0:
        new_rho = - (line[0])
        new_alpha = np.pi + line[1]
        return new_rho, new_alpha
    return line


def fix_hessian_form_vectorized(lines):
    """
        Fixe la forme de la représentation d'Hessain pour plusieurs droites à la fois.

        Args :
            lines (numpy.ndarray): tableau de shape (N, 2) représentant N droites, où chaque ligne est de la forme
                [rho, alpha], correspondant aux coordonnées polaires de la droite.

        Returns :
            numpy.ndarray: tableau de shape (N, 2) contenant les coordonnées polaires des droites modifiées.
    """
    lines = lines.copy()
    neg_rho_mask = lines[:, 0] < 0
    # On s'assure que les valeurs de rho sont positives
    lines[neg_rho_mask, 0] *= -1
    #On s'assure que toutes les lignes sont orientées dans la même orientation
    lines[neg_rho_mask, 1] -= np.pi
    return lines


def angle_diff(line1, line2):
    """
    Calcul de la différence angulaire entre deux droites représentées sous forme de coordonnées de Hesse.

    :param line1: Coordonnées de Hesse de la première droite sous forme de tuple (rho, alpha)
    :param line2: coordonnées de Hesse de la seconde droite sous forme de tuple (rho, alpha)
    :return: la différence angulaire entre les deux droites, en radians
    """
    diff = float('inf')
    if (line1[0] < 0) ^ (line2[0] < 0):
        if line1[0] < 0:
            diff = abs(fix_hessian_form(line1)[1] - line2[1]) % (np.pi)
        else:
            diff = abs(line1[1] - fix_hessian_form(line2)[1]) % (np.pi)

    diff = min(diff, abs(line1[1] - line2[1]) % np.pi)

    return diff


def angle_diff_vectorized(lines, line_to_calculate_diff):
    """
    Calcule la différence d'angle entre une ligne et un ensemble de lignes en utilisant la représentation d'Hesse.

    :param lines: Un tableau Numpy de shape (N, 2) contenant les N lignes représentées par leur distance à l'origine (rho) et leur angle (theta).
    :param line_to_calculate_diff: Une ligne dont on veut calculer la différence d'angle par rapport à toutes les lignes dans le tableau lines.
    :return: Un tableau Numpy de shape (N, 2) contenant les différences d'angle entre line_to_calculate_diff et chaque ligne dans lines.
    """
    hess_fixed_lines = fix_hessian_form_vectorized(lines)
    hess_fixed_calculate_line = fix_hessian_form(line_to_calculate_diff)

    diff_fixed = np.full(lines.shape[0], float('inf'))
    hess_test_mask = lines[:, 0] < 0

    if line_to_calculate_diff[0] >= 0:
        diff_fixed[hess_test_mask] = np.mod(np.abs(hess_fixed_lines[hess_test_mask, 1] - line_to_calculate_diff[1]),
                                            np.pi)
    else:
        diff_fixed[~hess_test_mask] = np.mod(lines[~hess_test_mask, 1] - hess_fixed_calculate_line[1], np.pi)

    diff_normal = np.mod(np.abs(lines[:, 1] - line_to_calculate_diff[1]), np.pi)

    return np.minimum(diff_normal, diff_fixed)


def cluster_mean_hessfixed(lines, cluster):
    """
    Calcule la moyenne des lignes d'un cluster donné, en utilisant la représentation d'Hesse corrigée si elle est plus
    cohérente avec les lignes du cluster.

    :param lines: Un tableau Numpy de shape (N, 2) contenant les N lignes représentées par leur distance à l'origine (rho) et leur angle (theta).
    :param cluster: Un ensemble d'indices de lignes appartenant au cluster.
    :return: La moyenne des lignes du cluster, en utilisant la représentation d'Hesse corrigée si elle est plus cohérente.
    """
    cluster_lines = lines[list(cluster)]
    normal_mean = np.mean(cluster_lines, axis=0)

    # On corrige les lignes calculées pour s'assurer qu'elles soient toutes représentées de la même manière
    hess_fixed_cluster_lines = fix_hessian_form_vectorized(cluster_lines)

    #On calcule l'orientation moyenne des droites des clusters
    hess_fixed_mean = np.mean(hess_fixed_cluster_lines, axis=0)

    normal_mean_diff = np.mean(angle_diff_vectorized(cluster_lines, normal_mean))
    hess_fixed_mean_diff = np.mean(angle_diff_vectorized(cluster_lines, hess_fixed_mean))

    return normal_mean if normal_mean_diff < hess_fixed_mean_diff else fix_hessian_form(hess_fixed_mean, True)


def get_all_line_intersection(lines, img_shape):
    """
        Calcule toutes les intersections entre les droites passant par les points spécifiés dans le tableau `lines`.
        Les intersections sont classées en trois groupes : intersections de droites parallèles, intersections de droites
        non parallèles et intersections hors champ de l'image.

        :param lines: Un tableau Numpy de shape (N, 2) contenant les N droites représentées par leur distance à l'origine (rho)
                      et leur angle (theta).
        :param img_shape: Un tuple représentant la taille de l'image sous forme (hauteur, largeur).

        :return: Trois éléments :
                 1. Un tableau Numpy de shape (M, 2) contenant les coordonnées de toutes les intersections trouvées entre les droites.
                 2. Une liste d'éléments de tuple contenant les indices des paires de droites de l'intersection.
                 3. Une liste d'ensembles contenant les indices des droites qui sont parallèles entre elles. Les ensembles sont
                    triés par taille décroissante, c'est-à-dire que les ensembles avec le plus de droites parallèles apparaissent
                    en premier.
    """
    parallel_sets_list = list() # Permet de stocker les positions des intersections
    intersections_info = list() # Permet de stocker les indices des paires de droite de l'intersection
    intersections = list() # Permet de strocker les groupes de droites parallèles
    for i, line in enumerate(lines):
        for j, line in enumerate(lines[i:], start=i):
            if i == j:
                continue
            # On calcule la position des intersections
            line_intersection = intersection(lines[i], lines[j], img_shape)
            # Si les coordonnées renvoyée par intersection() sont égales à '1, les droites ne sont pas sécantes
            if line_intersection[0] == -1 and line_intersection[1] == -1:
                set_exists = False
                # On vérifie si ces droites parallèles appartiennent à l'ensemble parallel_sets_list
                for next_set in parallel_sets_list:
                    # Dans le cas où les lignes i ou j apparitiennent à un ens de lignes parallèles, on ajoute i et j à l'ens
                    if (i in next_set) or (j in next_set):
                        set_exists = True
                        next_set.add(i) # ajoute i à l'ensemble
                        next_set.add(j) # ajoute j à l'ensemble
                        break
                # Sinon on cree un nouvel ensemble de lignes parallèles
                if not set_exists:
                    parallel_sets_list.append({i, j})
            else:
                # Si les droites sont sécantes, on vérifie que l'intersection se trouve dans l'image
                # On ignore les intersections situées dans l'image
                if not ((0 < line_intersection[0] < img_shape[0]) and (0 < line_intersection[1] < img_shape[1])):
                    intersections_info.append((i, j))
                    intersections.append(line_intersection)
    return intersections, intersections_info, sorted(parallel_sets_list, key=len, reverse=True)


def get_intersecting_line_clusters(intersections, intersections_info):
    """
        Calcule toutes les intersections entre les droites passant par les points spécifiés dans le tableau `lines`.
        Les intersections sont classées en trois groupes : intersections de droites parallèles, intersections de droites
        non parallèles et intersections hors champ de l'image.

        :param lines: Un tableau Numpy de shape (N, 2) contenant les N droites représentées par leur distance à l'origine (rho)
                      et leur angle (theta).
        :param img_shape: Un tuple représentant la taille de l'image sous forme (hauteur, largeur).

        :return: Trois éléments :
                 1. Un tableau Numpy de shape (M, 2) contenant les coordonnées de toutes les intersections trouvées entre les droites.
                 2. Une liste d'éléments de tuple contenant les indices des paires de droites de l'intersection.
                 3. Une liste d'ensembles contenant les indices des droites qui sont parallèles entre elles. Les ensembles sont
                    triés par taille décroissante, c'est-à-dire que les ensembles avec le plus de droites parallèles apparaissent
                    en premier.
    """
    # On applique DBSCAN afin de considéré uniquement les cluster de ligne
    dbscan_intersections = DBSCAN(eps=dbscan_eps_intersection_clustering, min_samples=8).fit(intersections) # 10, 8
    labels_intersections = dbscan_intersections.labels_

    # On récupère chaque ensemble de cluster (ensemble de paires de droite liée à un cluster)
    intersection_clusters = split_clusters_using_labels(intersections_info, labels_intersections)

    # On élimite les doublons de ligne dans le cluster
    unique_lines_each_cluster = list()
    for cluster in intersection_clusters:
        unique_lines = set()
        for lines in cluster:
            unique_lines.add(lines[0])
            unique_lines.add(lines[1])
        unique_lines_each_cluster.append(unique_lines)
    return sorted(unique_lines_each_cluster, key=len, reverse=True)


def get_parallel_line_clusters(lines, parallel_sets):
    """
     Identifies clusters of parallel lines.

         Parameters:
             - lines (np.ndarray): Array of line equations in Hessian normal form.
             - parallel_sets (List[set[int]]): List of sets containing indices of parallel lines.

         Returns:
             - List[set[int]]: List of sets containing indices of parallel line clusters.

         This function takes in an array of line equations in Hessian normal form and a list of sets containing indices of
         parallel lines. It identifies clusters of parallel lines by checking if the absolute difference between the mean
         y-coordinate of the sets is below a certain threshold. If the difference is below the threshold, the sets are merged
         and the mean y-coordinate is recalculated. This process is repeated until no more sets can be merged.
     """
    cur_sets = parallel_sets
    cur_means = list()
    # On calcule la moyenne de la coordonnée y de chaque ensemble de droites parallèles
    for next_set in cur_sets:
        cur_means.append(np.mean(lines[list(next_set)], axis=0)[1])

    i = 0
    while i < (len(cur_sets) - 1):
        for j in range(i + 1, len(cur_sets)):
            # Si la différence abs entre les moyennes de coordonnées y est inf à un seuil
            if abs(cur_means[i] - cur_means[j]) <parallel_angle_threshold:
                # On fusionne les ensembles
                cur_sets[i] = cur_sets[i] | cur_sets[j]
                # Puis on recalcule la moyenne de coordonnées y
                cur_means[i] = np.mean(lines[list(cur_sets[i])], axis=0)[1]
                cur_sets.pop(j)
                cur_means.pop(j)
                i = 0
                break
        i += 1
    return sorted(cur_sets, key=len, reverse=True)


def select_best_performing_cluster_pair(lines, intersections, parallel_sets):
    """
       Sélectionne la paire d'ensembles de droites qui fonctionne le mieux en termes de rectitude et d'orthogonalité.

       Parameters
       ----------
       lines : array_like
           Une liste de paires de points représentant les droites de l'image.
       intersections : array_like
           Une liste de paires de droites qui se croisent.
       parallel_sets : array_like
           Une liste d'ensembles de droites parallèles.

       Returns
       -------
       tuple
           Un tuple contenant les deux ensembles de droites choisis.

       """

    # Fusion des ensembles de droites qui se croisent avec ceux qui sont parallèles
    merged_clusters = intersections + parallel_sets
    # Calcul de la taille de chaque ensemble
    merged_sizes = list(map(lambda x: len(x), merged_clusters))

    pass_list = list()
    for i, cluster_i in enumerate(merged_clusters):
        for j, cluster_j in enumerate(merged_clusters[i:], start=i):
            if i == j:
                continue
            if angle_diff(cluster_mean_hessfixed(lines, cluster_i), cluster_mean_hessfixed(lines, cluster_j)) > two_line_cluster_threshold:
                pass_list.append((i, j))

    pass_list.sort(key = lambda x: (merged_sizes[x[0]] * merged_sizes[x[1]]), reverse=True)
    winner_pair = pass_list[0]

    return merged_clusters[winner_pair[0]], merged_clusters[winner_pair[1]]


def cluster_eliminate_duplicate(lines, cluster, intersect_line, img_shape):
    """
    Élimine les doublons de droites dans un cluster en utilisant une intersection donnée comme référence.

        :param lines: un tableau contenant les coordonnées des points de chaque droite.
        :type lines: numpy.ndarray

        :param cluster: un ensemble de paires de droites liées à un cluster.
        :type cluster: set

        :param intersect_line: une paire de points représentant la ligne d'intersection de deux droites.
        :type intersect_line: tuple

        :param img_shape: la taille de l'image en pixels (hauteur, largeur).
        :type img_shape: tuple

        :return: un ensemble de paires de droites uniques, sans doublons.
        :rtype: list
    """
    cluster_lines = lines[list(cluster)]
    intersection_points = list(map(lambda x: intersection(x, intersect_line, img_shape), cluster_lines))

    dbscan_test = DBSCAN(eps=dbscan_eps_duplicate_elimination, min_samples=1).fit(
        intersection_points)
    labels_test = dbscan_test.labels_

    merged_cluster = list()
    for i in range(max(labels_test) + 1):
        mask = (labels_test == i)
        merged_cluster.append(cluster_mean_hessfixed(lines, np.array(list(cluster))[mask]))

    return merged_cluster


def cluster_eliminate_non_matrix(merged_clusters, cluster_means, img_shape):
    """
    Élimine les ensembles de droites qui ne forment pas une matrice en sélectionnant les intersections les plus appropriées.

       :param merged_clusters: Un tuple contenant les deux ensembles de droites qui se croisent ou sont parallèles.
       :type merged_clusters: tuple
       :param cluster_means: Un tuple contenant les moyennes des deux ensembles de droites.
       :type cluster_means: tuple
       :param img_shape: La forme de l'image, i.e. (hauteur, largeur).
       :type img_shape: tuple
       :return: Un tuple contenant les ensembles de droites qui forment une matrice.
       :rtype: tuple
    """
    first_cluster, second_cluster = merged_clusters

    mean_first_cluster, mean_second_cluster = cluster_means

    intersections_first_cluster = list(map(lambda x: intersection(x, mean_second_cluster, img_shape), first_cluster))

    intersections_second_cluster = list(map(lambda x: intersection(x, mean_first_cluster, img_shape), second_cluster))

    best_intersections_first_cluster = select_nine_fittable_intersections(intersections_first_cluster)
    best_intersections_second_cluster = select_nine_fittable_intersections(intersections_second_cluster)

    return (np.array(first_cluster)[best_intersections_first_cluster],
            np.array(second_cluster)[best_intersections_second_cluster])


def select_nine_fittable_intersections(intersections):
    """
       Sélectionne les neuf meilleurs points d'intersection pour ajuster une matrice.

       Args :
           intersections : une liste de points d'intersection (tuple(x, y)).

       Returns:
           best_combination_indexes_reversed: une liste des neuf meilleurs index des points d'intersection fournis.
    """
    np_intersections = np.array(intersections)

    axis_variance = np.var(np_intersections, axis=0)

    metric_col = (0 if axis_variance[0] > axis_variance[1] else 1)
    metric_value = np_intersections[:, metric_col]

    sorted_idx = np.argsort(metric_value)
    metric_value = metric_value[sorted_idx]

    all_combinations_iter = itertools.combinations(np.array(list(enumerate(metric_value))), 9)
    all_combinations = np.stack(np.fromiter(all_combinations_iter, dtype=tuple))

    x = range(9)
    fitter = lambda y: np.poly1d(np.polyfit(x, y, polynomial_degree))(x)
    all_combinations_fitted_calculated = np.apply_along_axis(fitter, 1, all_combinations[:, :, 1])

    all_combinations_mse = (np.square(all_combinations[:, :, 1] - all_combinations_fitted_calculated)).mean(axis=1)

    best_combination_indexes = all_combinations[np.argmin(all_combinations_mse)][:, 0]

    sorted_idx_reverse_dict = {k: v for k, v in enumerate(sorted_idx)}

    best_combination_indexes_reversed = [sorted_idx_reverse_dict[k] for k in best_combination_indexes]

    return best_combination_indexes_reversed


def get_intersections_between_clusters(cluster1, cluster2, img_shape):
    """
       Calcule l'intersection de chaque ligne de `cluster1` avec chaque ligne de `cluster2`.

       Args:
           cluster1 (List[Tuple[int, int]]): Liste de tuples représentant des lignes, chaque tuple est de la forme `(y1, y2)`.
           cluster2 (List[Tuple[int, int]]): Liste de tuples représentant des lignes, chaque tuple est de la forme `(y1, y2)`.
           img_shape (Tuple[int, int]): Dimensions de l'image, sous forme de tuple `(hauteur, largeur)`.

       Returns:
           numpy.ndarray: Tableau numpy de dimensions (N, M, 2), où N est le nombre de lignes dans `cluster1`,
               M est le nombre de lignes dans `cluster2` et chaque élément du tableau est un tuple représentant
               les coordonnées x, y de l'intersection de deux lignes.
       """
    intersections = np.empty((len(cluster1), len(cluster2), 2), dtype=np.int32)
    for i, line_1 in enumerate(cluster1):
        for j, line_2 in enumerate(cluster2):
            intersections[i][j] = intersection(line_1, line_2, img_shape)
    return intersections


#=======================================================================================================================
#--------------------------------------------------Detection of sensors-------------------------------------------------
#=======================================================================================================================
line_amount = 200 # a#mount of best lines selected from hough lines
bilateral_filter_size = 7 # size of the bilateral filter for noise removal


parallel_threshold = 64 # intersections of 2 lines occured on distance > {}*image_size is assumed as parallel
parallel_angle_threshold = 0.04 # merge two parallel line clusters if angle difference is < {} (in radians)
two_line_cluster_threshold = 1.0 # angle difference between two line clusters of chess table should be < {} (in radians)


dbscan_eps_intersection_clustering = 10
dbscan_eps_duplicate_elimination = 3

polynomial_degree = 3  # used for fitting polynomial to intersection data, last step of the pipeline

###                        <<<<<<<<<<<<<<<<<<<<<<<<<<<<Main Params>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
params_Canny = (0, 40)
nbre_lines_selected = 150