import * as d3 from 'd3';

function polygonGenerator(communityGroup: Map<number, string[]>, groupId: number, nodesSelection: any) {
  // select nodes of the group, retrieve its positions and return the convex hull of the specified points
  const group: string[] = communityGroup.get(groupId)!;

  let node_coords = nodesSelection
    .data()
    .filter((d: any) => {
      return group.includes(d.name);
    })
    .map((d: any) => [d.x, d.y]);
  //  计算凸包时，多边形顶点数至少为3，若小于3，则补齐至3个
  if (node_coords.length === 1) {
    const gap = 20;
    node_coords = [
      [
        node_coords[0][0],
        node_coords[0][1] - gap,
      ], [
        node_coords[0][0] - gap / 2 * Math.sqrt(3),
        node_coords[0][1] + gap / 2,
      ], [
        node_coords[0][0] + gap / 2 * Math.sqrt(3),
        node_coords[0][1] + gap / 2,
      ],
    ];
  } else if (node_coords.length === 2) {
    const gap = 7;
    node_coords = [
      [
        node_coords[0][0] - gap,
        node_coords[0][1],
      ], [
        node_coords[0][0] + gap,
        node_coords[0][1],
      ], [
        node_coords[1][0] - gap,
        node_coords[1][1],
      ], [
        node_coords[1][0] + gap,
        node_coords[1][1],
      ],
    ]
  } else if (node_coords.length === 0) {
    return null;
  }

  return d3.polygonHull(node_coords);
};

export function getHullPaths(communityGroup: Map<number, string[]>, svgGroup: any) {
  const groupIds = [...communityGroup.keys()];
  console.log('groupIds', groupIds)
  const colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
  // ["#6b486b", "green", "blue", "#a05d56", "red", "#6b486b", "#a05d56", "#d0743c", "#ff8c00", "#6b486b", "#a05d56", "#d0743c", "#ff8c00", "#6b486b", "#a05d56", "#d0743c", "#ff8c00"]

  const paths = svgGroup
    .append('g')
    .attr('id', '#hull_layer')
    .selectAll("#hull_paths")
    .data(groupIds, (d: any) => +d)
    .enter()
    .append('g')
    .attr("id", '#hull_paths')
    .attr("fill-opacity", 0.1)
    .attr("stroke-opacity", 0.8)
    .append("path")
    .attr("stroke", (d: any) => {
      return colors[d];
    })
    .attr('stroke-width', 2)
    .attr ('pointer-events', 'visibleStroke')
    .on('mouseover', function() {
      d3.select(this).attr("stroke-width", 3)
        .style('cursor', 'pointer');
    })
    .on('mouseout', function() {
      d3.select(this).attr("stroke-width", 2)
        .style('cursor', 'default');
    })
    .attr("opacity", 1)
    .attr("fill", (d: any) => {
      return colors[d];
    });

  return paths;
}

export function updateGroups(communityGroup: Map<number, string[]>, paths: any, nodesSelection: any) {
  const groupIds = [...communityGroup.keys()];

  if (!groupIds.length)
    return;
  const margin: number = 1.2;
  const valueline = d3.line()
    .x((d: any) => d[0])
    .y((d: any) => d[1])
    .curve(d3['curveCatmullRomClosed'])


  groupIds.forEach(groupId => {
    let centroid: any[] = [];

    let path = paths.filter((d: any) => d === groupId)
      .attr('transform', 'translate(0,0) scale(1)')
      .attr('d', (d: any) => {
        const polygon = polygonGenerator(communityGroup, d, nodesSelection);
        if (!polygon) {
          return null;
        }
        centroid = d3.polygonCentroid(polygon);

        // to scale the shape properly around its points: move the 'g' element to the centroid point, translate
        // all the path around the center of the 'g' and then we can scale the 'g' element properly
        return valueline(polygon.map((point: any) => [point[0] - centroid[0], point[1] - centroid[1]]));
      });

    if (!centroid.length)
      return;
    d3.select(path.node().parentNode)
      .attr('transform', 'translate(' + centroid[0] + ',' + (centroid[1]) + ') scale(' + margin + ')');

  });
}