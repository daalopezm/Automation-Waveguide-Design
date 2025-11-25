import gdsfactory as gf
import numpy as np

@gf.cell
def trench(length=10, width=20, out_waveguide_support=2, width_waveguide_support=3, layer=(1, 0)) -> gf.Component:
    """Returns trench component.

    Args:
        length: trench length.
        width: trench width.
        depth: trench depth.
        layer: trench layer.

    """
    c = gf.Component()
    points = [(-width / 2, -length / 2), (-width / 2, length / 2), (width / 2, length / 2),
              (width / 2, length / 2-width_waveguide_support/2), (width / 2-out_waveguide_support, length / 2-width_waveguide_support/2),
              (width / 2-out_waveguide_support, -length / 2+width_waveguide_support/2),(width / 2, -length / 2+width_waveguide_support/2),
              (width / 2, -length / 2)]
    trench = c.add_polygon(points, layer=layer)
    c.add_port(name="t1", center=(-width / 2, 0), width=0.5, orientation=180, layer=layer, port_type="edge_coupler")
    c.add_port(name="t2", center=(width / 2-out_waveguide_support-4, 0), width=0.5, orientation=0, layer=layer, port_type="edge_coupler")
    return c

@gf.cell
def waveguide_bend(length=10, width=0.5, radius1=4, radius2=4, layer=(1, 0)) -> gf.Component:
    """Returns waveguide with trench component.

    Args:
        length: waveguide length.
        width: waveguide width.

    """
    c = gf.Component()
    p0 = gf.path.straight(length=length/16)
    p1 = gf.path.straight(length=length)
    p2 = gf.path.euler(radius=radius1, angle=75, p=0.5, use_eff=False)
    p3 = gf.path.euler(radius=radius2, angle=-75, p=0.5, use_eff=False)
    p4 = gf.path.straight(length=length/16)
    p = p0 + p1 + p2 + p3 + p4
    wg = c << gf.path.extrude(p, layer=layer, width=width)
    o1_center = tuple(np.array(p0.center) - np.array([length/32, 0]))
    o2_center = tuple(np.array(p4.center) + np.array([length/32, 0]))

    c.add_port(name="o1", center=o1_center, width=width, orientation=180, layer=layer)
    c.add_port(name="o2", center=o2_center, width=width, orientation=0,   layer=layer)
    return c



@gf.cell
def electrodes_x_cut(lenght=10, duty=0.4, period=0.2, prominence=1, metal_layer=(10,0), 
                     width_wg = 0.3, gap = 0.1, alpha = 0.5,
                     pad_top_width = 3, pad_buttom_width = 3,
                     pad_top_touch_width = 3, pad_buttom_touch_width = 3):
    n_electrodes = int(lenght//period)
    print(n_electrodes)
    c=gf.Component()

    # Top electrodes
    pad_top_touch = c.add_polygon([(lenght/2-2*pad_top_touch_width/2, prominence+width_wg/2+gap+pad_top_width), 
                                   (lenght/2-2*pad_top_touch_width/2, prominence+width_wg/2+gap+pad_top_width+pad_top_touch_width), 
                                   (lenght/2+2*pad_top_touch_width/2, prominence+width_wg/2+gap+pad_top_width+pad_top_touch_width), 
                                   (lenght/2+2*pad_top_touch_width/2, prominence+width_wg/2+gap+pad_top_width)], 
                                   layer=metal_layer)
    pad_top = c.add_polygon([(0, prominence+width_wg/2+gap), 
                             (0, prominence+width_wg/2+gap+pad_top_width), 
                             ((n_electrodes+duty)*period, prominence+width_wg/2+gap+pad_top_width), 
                             ((n_electrodes+duty)*period, prominence+width_wg/2+gap)], 
                             layer=metal_layer)
    for k in range(n_electrodes):
        x1, y1 = k*period, prominence+width_wg/2+gap
        x2, y2 = k*period+period*duty, prominence+width_wg/2+gap
        x3, y3 = k*period+period*duty, width_wg/2+gap
        x4, y4 = k*period, width_wg/2+gap

        pad = c.add_polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], layer=metal_layer)

    # Buttom electrodes
    pad_buttom_touch = c.add_polygon([(lenght/2-2*pad_buttom_touch_width/2, -prominence-width_wg/2-gap-pad_buttom_width), 
                                      (lenght/2-2*pad_buttom_touch_width/2, -prominence-width_wg/2-gap-pad_buttom_width-pad_buttom_touch_width), 
                                      (lenght/2+2*pad_buttom_touch_width/2, -prominence-width_wg/2-gap-pad_buttom_width-pad_buttom_touch_width), 
                                      (lenght/2+2*pad_buttom_touch_width/2, -prominence-width_wg/2-gap-pad_buttom_width)], 
                                      layer=metal_layer)
    pad_buttom = c.add_polygon([(0, -prominence-width_wg/2-gap), 
                                (0, -prominence-width_wg/2-gap-pad_buttom_width), 
                                ((n_electrodes+duty)*period, -prominence-width_wg/2-gap-pad_buttom_width), 
                                ((n_electrodes+duty)*period, -prominence-width_wg/2-gap)], 
                                layer=metal_layer)
    for k in range(n_electrodes):
        x1, y1 = k*period, -prominence/2-width_wg/2-gap
        x2, y2 = k*period+period*duty/2-period*duty*alpha/2, -width_wg/2-gap
        x3, y3 = k*period+period*duty/2+period*duty*alpha/2, -width_wg/2-gap
        x4, y4 = k*period+period*duty, -prominence/2-width_wg/2-gap  
        x5, y5 = k*period+period*duty, -prominence-width_wg/2-gap 
        x6, y6 = k*period, -prominence-width_wg/2-gap  

        pad = c.add_polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)], layer=metal_layer)

    return c





@gf.cell
def marker(layer=(2, 0)) -> gf.Component:
    """Returns marker component.

    Args:
        length: marker length.
        width: marker width.
        layer: marker layer.

    """
    length=1.5
    width=1.5
    l = gf.Component()
    points1 = [(-4*width, -4*length), (-4*width, 0), (0, 0), (0, -4*length)]
    square1 = l.add_polygon(points1, layer=layer)
    points2 = [(4*width, 4*length), (4*width, 0), (0, 0), (0, 4*length)]
    square2 = l.add_polygon(points2, layer=layer)

    separation = 8.5
    a = l << gf.components.rectangle(size=(80, 2*width), layer=layer)
    a = a.move((separation, -width))
    a.irotate(0, center=(0, 0))
    b = l << gf.components.rectangle(size=(80, 2*width), layer=layer)
    b = b.move((separation, -width))
    b.irotate(1, center=(0, 0))
    c = l << gf.components.rectangle(size=(80, 2*width), layer=layer)
    c = c.move((separation, -width))
    c.irotate(2, center=(0, 0))
    d = l << gf.components.rectangle(size=(80, 2*width), layer=layer)
    d = d.move((separation, -width))
    d.irotate(3, center=(0, 0))
    return l