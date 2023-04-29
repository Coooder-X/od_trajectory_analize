<template>
  <div></div>
</template>

<script lang="ts">
import { computed, defineComponent, onBeforeMount, onMounted, watch, Ref, ref, onBeforeUnmount } from "vue";
import * as d3 from "d3";

export default defineComponent({
  name: "PolarHeatMap",
  components: {},
  props: {
    svg: {
      type: Object,
      required: true,
    },
  },
  emits: ["closeMenu"],
  setup(props, contex) {
    const heatMapGroup: Ref<any> = ref(null);

    // 定义颜色比例尺
    const color = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]); // 假设数据的值在0到1之间

    // 定义角度比例尺
    const angle = d3
      .scaleLinear()
      .domain([0, 23]) // 假设数据有24个角度
      .range([0, 2 * Math.PI]); // 将数据的角度映射到0到2π之间

    // 定义半径比例尺
    const radius = d3
      .scaleLinear()
      .domain([0, 3]) // 假设数据有3个半径
      .range([20, 70]); // 将数据的半径映射到props.svg的半径

    // 创建一个弧形生成器
    const arc = d3
      .arc()
      .innerRadius(function (d: any) {
        return radius(d.r);
      }) // 设置内半径为数据的半径
      .outerRadius(function (d: any) {
        return radius(d.r + 1);
      }) // 设置外半径为数据的半径加1
      .startAngle(function (d: any) {
        return angle(d.a);
      }) // 设置起始角度为数据的角度
      .endAngle(function (d: any) {
        return angle(d.a + 1);
      }); // 设置终止角度为数据的角度加1

    onMounted(() => {

      // 创建一个分组元素，并平移到props.svg的中心
      heatMapGroup.value = props.svg
        .append("g")
        .attr('id', 'heatMapGroup')

      console.log("mounted", heatMapGroup.value);

      // 准备一些随机数据，每个元素包含一个角度、一个半径和一个值
      const data = [];
      for (let i = 0; i < 24; i++) {
        for (let j = 0; j < 3; j++) {
          data.push({ a: i, r: j, v: Math.random() });
        }
      }

      // 为每个数据元素添加一个路径元素，并设置其属性
      heatMapGroup.value.selectAll("path")
        .data(data)
        .enter()
        .append("path")
        .attr("d", arc) // 使用弧形生成器绘制路径
        .attr("fill", function (d: any) {
          return color(d.v);
        }) // 使用颜色比例尺设置填充颜色
        .attr('opacity', 0.7)
        .attr("stroke", "white"); // 设置描边颜色为白色

      // 添加一个标题
      props.svg
        .append("text")
        .attr("x", 0)
        .attr("y", 0 - 100)
        .attr("text-anchor", "middle")
        .attr("font-size", "18px")
        .text("极坐标热力图示例");

      // 添加一个图例
      // const legend = props.svg
      //   .append("g")
      //   .attr(
      //     "transform",
      //     "translate(" +
      //       (width - margin.right - 20) +
      //       "," +
      //       (height / 2 - 100) +
      //       ")"
      //   );

      // legend
      //   .append("text")
      //   .attr("x", 0)
      //   .attr("y", -10)
      //   .attr("text-anchor", "middle")
      //   .text("值");

      const defs = props.svg.append("defs");

      const linearGradient = defs
        .append("linearGradient")
        .attr("id", "linear-gradient");

      linearGradient.selectAll("stop").data(
        color.ticks().map(function (t: any, i: number, n: number[]) {
          return { offset: i / n.length, color: color(t) };
        })
      );

      // 创建一个径向轴生成器
      // const radialAxis = d3
      //   .axisBottom(radius)
      //   .ticks(3) // 设置刻度值为1-3
      //   .tickFormat(function (d: any) {
      //     return d;
      //   }); // 设置刻度格式为数字

      // // 创建一个切向轴生成器
      // const tangentialAxis = d3
      //   .axisLeft(angle)
      //   .ticks(24) // 设置刻度值为0-23
      //   .tickFormat(function (d: any) {
      //     return d;
      //   }); // 设置刻度格式为数字

      // 创建一个分组元素，用来存放切向轴的标签
      const labelGroup = heatMapGroup.value.append("g");
      // .attr("class", "labels");

      console.log(angle.domain());
      let timeData = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23,
      ];
      // 为每个角度创建一个文本元素
      labelGroup
        .selectAll("text")
        .data(timeData) // 使用角度比例尺的定义域作为数据
        .enter()
        .append("text")
        .attr("dominant-baseline", "central")
        .attr("x", function (d: any) {
          return Math.cos(angle(d) - Math.PI / 2) * (radius(3) + 10);
        }) // 设置x坐标为角度对应的余弦值乘以半径加10
        .attr("y", function (d: any) {
          return Math.sin(angle(d) - Math.PI / 2) * (radius(3) + 10);
        }) // 设置y坐标为角度对应的正弦值乘以半径加10
        .attr("text-anchor", "middle") // 设置文本对齐方式为居中
        .attr("font-size", "12px") // 设置字体大小为12像素
        .text(function (d: any) {
          return d;
        }); // 设置文本内容为数据值
    });

    onBeforeUnmount(() => {
      //  销毁组件时删除这个图表的svg
      d3.select('#heatMapGroup').remove();
      d3.selectAll('text').remove();
    });

    watch(props.svg, () => {
      if (props.svg) console.log("svg", props.svg);
    });

    return {};
  },
});
</script>

<style scoped>
</style>
