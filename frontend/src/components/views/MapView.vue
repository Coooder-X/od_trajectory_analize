<template>
  <div class="map-view">
    <view-header viewId="B" title="地图视图"></view-header>
    <div class="map-view-content">
      <div class="side-bar">
        <div class="side-bar-button" @click="toggleClusterLayer($event)">
          <el-tooltip
            :content="clusterLayerShow? '隐藏OD点' : '显示OD点'"
            placement="right">
            <img src="@/assets/location_fill.svg" alt="" />
          </el-tooltip>
        </div>
        <el-popover :visible="clusteringConfigVisible" placement="right" :width="200">
          <template #reference>
            <div class="side-bar-button" @click="()=>{clusteringConfigVisible = !clusteringConfigVisible}">
              <el-tooltip
                content="聚类选项"
                placement="right">
                <img src="@/assets/chart-bubble.svg" alt="" />
              </el-tooltip>
            </div>
          </template>
          <div style="text-align: center; margin: 0">
            <b style="margin-bottom: 5px;">聚类参数配置</b>
            <div class="clustering-config-row">
              <span class="config-line">k:</span>
              <el-input v-model="k" class="config-input"></el-input>
            </div>
            <div class="clustering-config-row">
              <span class="config-line">θ:</span>
              <el-input v-model="theta" class="config-input"></el-input>
            </div>
            <el-button type="primary" @click="doClustering">聚类</el-button>
          </div>
        </el-popover>
        <div class="side-bar-button" @click="modifyMode(MapMode.SELECT)">
          <el-tooltip
            content="刷选"
            placement="right">
            <img src="@/assets/框选.svg" alt="" />
          </el-tooltip>
        </div>
        <div class="side-bar-button">
          <el-tooltip
            content="点选"
            placement="right">
            <img src="@/assets/pointer.svg" alt="" />
          </el-tooltip>
        </div><div class="side-bar-button">
          <el-tooltip
            content="显示OD流"
            placement="right">
            <img src="@/assets/pin-distance-line-active.svg" alt="" />
          </el-tooltip>
        </div>
        <div class="side-bar-button">
          <el-tooltip
            content="显示轨迹"
            placement="right">
            <img src="@/assets/路径分析.svg" alt="" />
          </el-tooltip>
        </div>
        <div class="side-bar-button">
          <el-tooltip
            content="过滤"
            placement="right">
            <img src="@/assets/filter.svg" alt="" />
          </el-tooltip>
        </div>
      </div>
      <map-comp class="map-comp"></map-comp>
    </div>
  </div>
</template>

<script lang='ts'>
/* eslint-disable */
import { defineComponent, computed, onMounted, ref, Ref } from "vue";
import { useStore } from 'vuex';
import ViewHeader from "../ViewHeader.vue";
import MapComp from "../MapComp.vue";
import { MapMode } from '@/map-interface'
import {
  Document,
  Menu as IconMenu,
  Location,
  Setting,
} from "@element-plus/icons-vue";

export default defineComponent({
  components: {
    MapComp,
    ViewHeader,
    Document,
    IconMenu,
    Location,
    Setting,
  },
  name: "MapView",
  props: {},
  setup() {
    const store = useStore();
    const { getters } = store;
    const clusterLayerShow = computed(() => store.state.layers.clusterLayerShow);
    const clusteringConfigVisible: Ref<boolean> = ref(false);
    const k: Ref<number> = ref(25);
    const theta: Ref<number> = ref(50);

    onMounted(() => {
      store.dispatch('helloWorld');
    })

    const chooseMode = (index: number) => {
      console.log("chooseMode", index);
    };

    // 切换是否显示轨迹点图层
    const toggleClusterLayer = () => {
      store.commit('setClusterLayerShow', !store.state.layers.clusterLayerShow);
    }

    const doClustering = () => {
      console.log('doClustering')
      const [startHour, endHour] = getters.timeScope;
      const [startDay, endDay] = getters.dateScope;
      clusteringConfigVisible.value = !clusteringConfigVisible.value
      store.dispatch('getClusteringResult', {params: {k: k.value, theta: theta.value, startDay: startDay + 1, endDay: endDay + 1, startHour, endHour}});
      modifyMode(MapMode.CLUSTERED);
    }

    const modifyMode = (mode: string) => {
      store.commit('toggleMapMode', mode);
    }

    return {
      clusterLayerShow,
      toggleClusterLayer,
      modifyMode,
      k,
      theta,
      doClustering,
      clusteringConfigVisible,
      MapMode,
    };
  },
});
</script>

<style scoped>
img {
  height: 30px;
  width: 30px;
}
img:hover {
  fill: aqua;
}
.map-view {
  --menu-width: 50px;
  height: 500px;
  width: 850px;
  background-color: white;
}

.map-view-content {
  height: calc(100% - var(--header-height));
  display: flex;
}

.side-bar {
  padding-top: 5px;
  height: calc(100% - var(--header-height));
  width: var(--menu-width);
  position: absolute;
}

.side-bar-button {
  display: flex;
  height: 50px;
  align-items: center;
  justify-content: center;
}
.side-bar-button:hover {
  cursor: pointer;
  background-color: aliceblue;
}

.map-comp {
  margin-left: var(--menu-width);
  height: 100%;
  width: calc(100% - var(--menu-width));
}

.clustering-config-row {
  display: flex;
  justify-content: center;
  align-content: center;
  margin-bottom: 10px;
}

.config-line {
  line-height: 30px;
  margin-right: 10px;
}

.config-input {
  width: 150px;
}
</style>
