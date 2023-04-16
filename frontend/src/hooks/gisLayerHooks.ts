import { computed, ComputedRef, Ref, ref, watch } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
import { colorTable } from '@/color-pool';

export function useBrush({
  clusterLayerSvg, odPoints, project
}: {
  clusterLayerSvg: Ref<any | null>, odPoints: ComputedRef<number[][]>, project: Function
}) {
  const store = useStore();
  const { getters } = store;
  const mapMode = getters.mapMode;
  let pointClusterMap = computed(() => getters.pointClusterMap);
  let odIndexList = computed(() => getters.odIndexList);
  const x0 = ref(0);
  const y0 = ref(0);
  const x1 = ref(0);
  const y1 = ref(0);
  console.log('clusterLayerSvg', clusterLayerSvg)
  let odCircles: any
  let brush: any
  let brushLayer: any

  // watch(clusterLayerSvg, (newValue, oldValue) => {
  //   if(newValue && !oldValue) {
  //     console.log('画布svg第一次初始化')
  //     brushLayer = clusterLayerSvg.value.append("g")
  //     brushLayer
  //       .attr("class", "brush")
  //       .call(brush)
  //       .style('display', 'none')
  //   }
  // })

  watch(clusterLayerSvg, () => {
    if(clusterLayerSvg.value) {
      // console.log([clusterLayerSvg.value.attr('width'), clusterLayerSvg.value.attr('height')])
      odCircles = clusterLayerSvg.value.selectAll("circle")
      brush = d3.brush()
        .extent([[0, 0], [800, 465]])  //  [clusterLayerSvg.value.attr('width'), clusterLayerSvg.value.attr('height')])
        .on("brush", brushed); // 设置刷取事件的回调函数
    
      // 在画布上添加刷取元素
      brushLayer = clusterLayerSvg.value.append("g")
      brushLayer
        .attr("class", "brush")
        .call(brush)
        .style('display', 'none')
    }
  }, {deep: false});

  // 定义刷取事件的回调函数
  function brushed(event: any) {
    odCircles = clusterLayerSvg.value.selectAll("circle")
    // 获取刷取范围的坐标
    const selection = event.selection;
    console.log(selection)
    x0.value = selection[0][0];
    y0.value = selection[0][1];
    x1.value = selection[1][0];
    y1.value = selection[1][1];

    // 遍历所有的 circle 元素
    odCircles.each(function (d: any, i: number) {
      // 获取当前 circle 的坐标
      const cx = project(d).x;
      const cy = project(d).y;

      // 判断当前 circle 是否在刷取范围内
      const inside = cx >= x0.value && cx <= x1.value && cy >= y0.value && cy <= y1.value;

      // 根据判断结果改变当前 circle 的颜色
      if (inside) {
        d3.select(this).style("fill", function(point: number[]) {
          const index = odIndexList.value[i]
          if(pointClusterMap.value.has(index))
            return colorTable[pointClusterMap.value.get(index)];
          return "#ff3636"
        });
      } else {
        d3.select(this).style("fill", "lightblue");
      }
    });
  }

  function setBrushLayerVisible(value: boolean) {
    console.log('setBrushLayerVisible', value)
    brushLayer
      .style('display', value ? '' : 'none');
  }

  const selectedPoints = computed(() => {
    return odCircles.filter(function (d: any) {
      // 获取当前 circle 的坐标
      const cx = d.x;
      const cy = d.y;
      // 判断当前 circle 是否在刷取范围内
      return cx >= x0.value && cx <= x1.value && cy >= y0.value && cy <= y1.value;
    });
  });

  // function getSelectedPoints() {
  //   return odCircles.filter(function (d: any) {
  //     // 获取当前 circle 的坐标
  //     const cx = d.x;
  //     const cy = d.y;
  //     // 判断当前 circle 是否在刷取范围内
  //     return cx >= x0.value && cx <= x1.value && cy >= y0.value && cy <= y1.value;
  //   });
  // }

  return {
    setBrushLayerVisible,
    selectedPoints,
  }
}
