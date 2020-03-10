"""
功能：在特定需求下实现曲面的绘制并测算距离
作者：李腾耀
日期：2019/07/07
+ 已经进行所有计算项的核对
+ 所有度量值设定请使用浮点数
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties


def cal_angle(vx, vy):
    """
    计算两个向量间的夹角（基于余弦公式）
    :param vx: 向量x
    :param vy: 向量y
    :return: 向量夹角（弧度制）
    """
    return np.arccos(np.dot(vx, vy) / (np.sqrt(np.dot(vx, vx)) * np.sqrt(np.dot(vy, vy))))


def index2angle(index, angle_scope=np.pi / 3, surface_density=360):
    """
    对于圆柱面在xz轴上的投影，完成数值数组下标向对应角度的转换
    :param index: 数组索引（整数类型）
    :param angle_scope: 弧度范围的一半（由于选取范围为[-angle_scope, angle_scope]）
    :param surface_density: 圆柱面投影的切分密度
    :return: 对应弧度值（从最左侧连边开始算起，如果超过angle_scope，说明到了中心线的另一侧）
    """

    return angle_scope * 2 / surface_density * index


def angle2index(angle, angle_scope=np.pi / 3, surface_density=360):
    """
    对于圆柱面在xz轴上的投影，完成角度向对应数值数组下标的转换
    :param angle: 节点在投影上所处的角度（从最左侧连边开始算起）
    :param angle_scope: 弧度范围的一半（由于选取范围为[-angle_scope, angle_scope]）
    :param surface_density: 圆柱面投影的切分密度
    :return: 对应数组下标索引（必须为整数值）
    """

    return int(angle / (angle_scope * 2 / surface_density))


if __name__ == "__main__":

    """
    主要绘图和计算脚本，根据后期功能集成需求，可以转换为多个函数或构建类对象
    """

    generatrix = 2.0  # 圆柱面母线长度

    # 创建绘图实例
    fig = plt.figure()
    ax = Axes3D(fig)

    # -----------------绘制圆柱面------------------------#
    # 对平面进行划分
    surface_density = 360  # 圆柱面底面圆划分为360份
    height_density = 30  # 圆柱面高度方向划分为30份
    radius_cylinder = 1.0  # 圆柱面底面圆的半径
    angle_scope = np.pi / 3  # 圆柱面角度的一半（特别注意：圆柱面角度是按照中心线两侧角度的2倍进行设定的）
    theta = np.linspace(-angle_scope, angle_scope, surface_density)  # 圆柱面底面角度的取值范围：0-pi/2
    height = np.linspace(-generatrix / 2, generatrix / 2, height_density)  # 圆柱面高度方向的取值范围，默认与母线长度保持一致
    # 构建圆柱面上的点，自底面向上逐层构建
    cx = np.outer(radius_cylinder * np.sin(theta), np.ones(len(height)))
    cy = np.outer(np.ones(len(theta)), height)
    cz = -np.outer(radius_cylinder * np.cos(theta) - radius_cylinder, np.ones(len(height)))
    # 绘制圆柱面
    # 配色参考：https://matplotlib.org/examples/color/colormaps_reference.html
    ax.plot_surface(cx, cy, cz, cmap=plt.get_cmap("rainbow"))

    # -----------------绘制圆柱面切面平面------------------------#
    # 计算投影面上圆柱底面切线的斜率和切点
    slope = (cz[-1][0] - cz[0][0]) / (cx[-1][0] - cx[0][0])
    x_m = cx[int(len(cx) / 2)][0]  # x_m = 0.0
    z_m = cz[int(len(cz) / 2)][0]  # z_m = 0.0
    # 获取切面坐标点集
    px = np.linspace(cx[0][0], cx[-1][0], height_density)
    py = height
    px_ext, py_ext = np.meshgrid(px, py)
    pz_ext = slope * px_ext + (z_m - slope * x_m) * np.ones_like(px_ext)
    # 绘制切面平面
    # 配色参考：https://matplotlib.org/examples/color/colormaps_reference.html
    ax.plot_surface(px_ext, py_ext, pz_ext, cmap=plt.get_cmap("ocean"))

    # -----------------绘制球面------------------------#
    # 设定球心位置和坐标
    circle_height = generatrix / 2
    radius_circle = 0.3
    point_center = np.array([0, 0, radius_cylinder])  # 与起始坐标有关，与圆柱面对应底面圆在纵向映射方向重合

    # 获取球面点集
    alpha = np.linspace(0, 2 * np.pi, surface_density)
    gamma = np.linspace(0, np.pi, surface_density)
    sx = radius_circle * np.outer(np.cos(alpha), np.sin(gamma)) + point_center[0]
    sz = radius_circle * np.outer(np.sin(alpha), np.sin(gamma)) + point_center[2]
    sy = radius_circle * np.outer(np.ones(len(alpha)), np.cos(gamma)) + point_center[1]
    # 绘制球面
    # 配色参考：https://matplotlib.org/examples/color/colormaps_reference.html
    ax.plot_surface(sx, sy, sz, cmap=plt.get_cmap("rainbow"))

    # -----------------计算圆柱面一点到球面的距离------------------------#
    height_selected = 0.3  # 选择该点的高度
    index_surface = angle2index(index2angle(30))  # 可以使用角度和坐标索引转换函数，自由切换指定节点位置
    index_height = int(np.floor(height_selected * height_density / generatrix))
    point_cylinder = np.array([cx[index_surface][0], cy[0][index_height], cz[index_surface][0]])
    # 根据欧拉公式，计算距离
    distance = np.sqrt(np.sum(np.square(point_cylinder - point_center))) - radius_circle
    print("相对于球面的距离为: %f 单位距离" % distance)
    ax.text((point_center[0] + point_cylinder[0]) / 2, (point_center[1] + point_cylinder[1]) / 2 + 0.02,
            (point_center[2] + point_cylinder[2]) / 2, "Distance: %.2f" % (distance + radius_circle))

    # -----------------计算圆柱面一点与球心的连线与球心垂线的夹角------------------------#
    vector_cylinder = point_cylinder - point_center
    point_center_projection = point_center.copy()  # bug修复：避免直接在引用上进行直接修改
    point_center_projection[1] = generatrix / 2.0
    vector_vertical_line = point_center_projection - point_center

    # 绘制辅助线
    point_line = np.hstack((point_cylinder[:, np.newaxis], point_center[:, np.newaxis]))
    ax.plot(point_line[0], point_line[1], point_line[2], linestyle="-", marker="o")
    point_ref_line = np.hstack(
        ([[point_center_projection[0]], [-point_center_projection[1]], [point_center_projection[2]]],
         point_center_projection[:, np.newaxis]))
    ax.plot(point_ref_line[0], point_ref_line[1], point_ref_line[2], linestyle=':')
    # 计算角度
    angle = cal_angle(vector_cylinder, vector_vertical_line)
    print("相对于球面垂线的夹角为: %f 度" % (angle / np.pi * 180))
    # ax.text(point_center[0], point_center[1] - 0.1, point_center[2], "angle: %.2f" % angle)

    # -----------------计算圆柱面上相同高度的两点之间的弧长------------------------#
    height_selected_b = height_selected  # 保证两个节点处于同一高度
    index_surface_b = angle2index(index2angle(300))  # 可以使用角度和坐标索引转换函数，自由切换指定节点位置
    index_height_b = int(np.floor(height_selected_b * height_density / generatrix))
    point_cylinder_a = point_cylinder.copy()  # 这个参考点可以另行选取，为了方便，我们使用之前的点point_cylinder
    point_cylinder_b = np.array([cx[index_surface_b][0], cy[0][index_height_b], cz[index_surface_b][0]])
    point_center_a = point_center.copy()  # 计算弧长对应的圆心位置
    point_center_a[1] = height_selected + py[0]  # 从平面底部开始计算y方向位置

    # 绘制节点和参考线
    ax.plot(cx[index_surface:index_surface_b, 0].flatten(),
            cy[index_surface:index_surface_b, index_height_b].flatten(),
            cz[index_surface:index_surface_b, 0].flatten())
    point_ref_line_a = np.hstack((point_cylinder_a[:, np.newaxis], point_center_a[:, np.newaxis]))
    ax.plot(point_ref_line_a[0], point_ref_line_a[1], point_ref_line_a[2], linestyle="--", marker="o")
    point_ref_line_b = np.hstack((point_cylinder_b[:, np.newaxis], point_center_a[:, np.newaxis]))
    ax.plot(point_ref_line_b[0], point_ref_line_b[1], point_ref_line_b[2], linestyle="--", marker="o")

    # 计算弧长
    vector_cylinder_a = point_cylinder_a - point_center_a
    vector_cylinder_b = point_cylinder_b - point_center_a
    angle_a = cal_angle(vector_cylinder_a, vector_cylinder_b)
    arc_len = angle_a * radius_cylinder

    print("相邻两点间的弧长为： %f 单位距离" % arc_len)
    ax.text((point_cylinder_b[0] + point_cylinder_a[0]) / 2, (point_cylinder_b[1] + point_cylinder_a[1]) / 2,
            (point_cylinder_b[2] + point_cylinder_a[2]) / 2,
            "Length of Arc: %.2f" % arc_len)

    # -----------------计算圆柱面上一点到对应切面的距离------------------------#
    point_cylinder_t = point_cylinder_b.copy()
    height_t = point_cylinder_t[1] + py[0]  # 高度从平面底部开始算起
    point_center_t = np.array([point_center[0], height_t, point_center[2]])
    vector_cylinder_t = point_cylinder_t - point_center_t
    point_ref = np.array([x_m, height_t, z_m])
    vector_ref = point_ref - point_center_t
    # 计算与中心垂线向量的夹角
    theta_t = cal_angle(vector_cylinder_t, vector_ref)
    distance_x = 0.0
    # 判定夹角是锐角还是钝角，计算公式有所不同
    if theta_t < np.pi / 2:
        distance_t = radius_cylinder - radius_cylinder * np.cos(theta_t)
    else:
        distance_t = radius_cylinder * np.sin(theta_t - np.pi / 2) + radius_cylinder
    print("点(%f,%f,%f)到切面的距离为：%f" % (point_cylinder_t[0], point_cylinder_t[1], point_cylinder_t[2], distance_t))

    # 绘制参考线
    # 计算平面上垂点的坐标
    ref_vx = point_cylinder_t[0]
    ref_vy = point_cylinder_t[1]
    ref_vz = 0.0
    ref_vertical_line = np.hstack((point_cylinder_t[:, np.newaxis], [[ref_vx], [ref_vy], [ref_vz]]))
    ax.plot(ref_vertical_line[0], ref_vertical_line[1], ref_vertical_line[2], linestyle="--", marker="o", linewidth=1)
    ax.text(ref_vx, ref_vy, ref_vz, "Vertical Distance: %.2f" % distance_t)

    ax.set_xlabel("x")  # 显示x坐标轴名称
    ax.set_ylabel("y")  # 显示y坐标轴名称
    ax.set_ylim(-1.0, 1.0)  # 限制y坐标轴的范围
    ax.set_zlabel("z")  # 显示z坐标轴名称

    # font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)  # 提供图片中的中文字符支持
    # ax.set_title("相位补偿参考示意图", fontproperties=font_set)  # 设定图的名称

    plt.savefig("fig.pdf", format="pdf")
    plt.savefig("fig.png", format="png")
    plt.savefig("fig.eps", format="eps")

    plt.show()
