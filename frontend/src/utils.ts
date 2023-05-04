import { ForceNode } from "./global-interface";
import * as d3 from 'd3';
import mapboxgl from "mapbox-gl";

//  返回修改后的 forceNodes，加入 avgLen 属性，代表OD簇中心之间的距离，与力导向节点的半径成正比。
export function calLenColor(forceNodes: ForceNode, cidCenterMap: Map<number, [number, number]>, map: any) {
  const project = (d: [number, number]) => {
    return map.project(new mapboxgl.LngLat(d[0], d[1]));
  }
  let min_len = 9999999, max_len = -1;
  const nodeLenMap: Map<string, number> = new Map();
  forceNodes.forEach((node: any) => {
    //  计算这对 OD 对之间，轨迹的数量
    const {name} = node;
    const [srcCid, tgtCid] = name.split('_').map(Number);
    const srcCoord = project(cidCenterMap.get(srcCid)!);
    const tgtCoord = project(cidCenterMap.get(tgtCid)!);
    const len = Math.round(Math.sqrt((srcCoord.x - tgtCoord.x) ** 2 + (srcCoord.y - tgtCoord.y) ** 2));
    
    min_len = Math.min(min_len, len);
    max_len = Math.max(max_len, len);
    nodeLenMap.set(name, len);
  });

  //  最小半径4，最大9，变化范围5
  const range = 5, base = 4;
  forceNodes.forEach((node: any) => {
    let value = nodeLenMap.get(node.name)!;
    value = base + range * (value - min_len) / (max_len - min_len);
    node.avgLen = value;
  });

  return forceNodes;
}

//  返回取色map，通过力导向节点的 name 获取对应颜色。颜色代表OD对内轨迹数，即OD对的热度。渐变为 蓝-白-红，从低到高。
export function calNodeColor(forceNodes: ForceNode, clusterPointMap: Map<number, number[]>, odPoints: Array<number[]>) {
  console.log('total len', odPoints.length)
  let min_num = 9999999, max_num = -1;
  const nodeNumMap: Map<string, number> = new Map();
  const nodeColorMap: Map<string, string> = new Map();
  forceNodes.forEach((node: any) => {
    //  计算这对 OD 对之间，轨迹的数量
    const {name} = node;
    const [srcCid, tgtCid] = name.split('_').map(Number);
    const [srcCluster, tgtCluster] = [clusterPointMap.get(srcCid)!, clusterPointMap.get(tgtCid)!];
    let cnt = 0;
    // console.log(srcCluster.length, tgtCluster.length)
    for (const srcP of srcCluster) {
      const oP = odPoints[srcP];
      for (const tgtP of tgtCluster) {
        const dP = odPoints[tgtP];
        if (oP[3] === dP[3] && oP[4] === 0 && dP[4] === 1) {  //  这两个 OD 点轨迹 ID 相同，属于同一条轨迹。符合方向，则计数
          cnt++;
          break;
        }
      }
    }
    min_num = Math.min(min_num, cnt);
    max_num = Math.max(max_num, cnt);
    nodeNumMap.set(name, cnt);
    node.trjNum = cnt;
    // console.log('cnt =', cnt)
  });

  // 创建一个线性比例尺
  const nodeColorPicker = d3.scaleLinear()
    // .domain([min_num - 1, max_num + 1]) // 数值范围
    // .range(['rgb(247, 247, 233)', '#ff4c4c']);
    .domain([min_num, (max_num + min_num) / 2, max_num]) // 数值范围
    // .range(["white", "salmon"]); // 颜色范围
    // .range([d3.rgb(0, 136, 255).toString(), d3.rgb(255,255,255).toString(), d3.rgb(227, 0, 0).toString()]); // 颜色范围
    // .range([d3.rgb(80, 122, 175).toString(), d3.rgb(247, 247, 233).toString(), d3.rgb(190,92,55).toString()]); // 颜色范围
    .range(['rgb(80, 122, 175)', 'rgb(247, 247, 233)', '#ff4c4c']);
    // .range(["#375093", "#83A121"]); // 颜色范围

  //  归一化，分配颜色
  const range = max_num - min_num;
  nodeNumMap.forEach((value: number, key: string) => {
    nodeColorMap.set(key, nodeColorPicker(value));
    value = range * (value - min_num) / (max_num - min_num);
  });

  return nodeColorMap;
}