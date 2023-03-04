<template>
  <div class="map-view">
    <view-header viewId="A" title="地图视图"></view-header>
    <div class="map-view-content">
      <el-menu
        default-active="2"
        class="el-menu-vertical"
        @select="chooseMode"
      >
        <el-menu-item :index="MapMode.SHOW_POINTS">
          <el-tooltip
            :content="MapModeTooltip[MapMode.SHOW_POINTS]"
            placement="right"
            >
            <el-icon><icon-menu /></el-icon>
          </el-tooltip>
        </el-menu-item>
        <el-menu-item :index="MapMode.SELECT">
          <el-tooltip
            :content="MapModeTooltip[MapMode.SELECT]"
            placement="right"
            >
            <el-icon><setting /></el-icon>
          </el-tooltip>
        </el-menu-item>
      </el-menu>
      <map-comp class="map-comp"></map-comp>
    </div>
  </div>
</template>

<script>
/* eslint-disable */
import { defineComponent } from "vue";
import ViewHeader from "../ViewHeader";
import MapComp from "../MapComp";
import { MapMode, MapModeTooltip } from '@/map-interface.ts'
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
    const chooseMode = (index) => {
      console.log("chooseMode", index);
    };

    return {
      chooseMode,
      MapMode,
      MapModeTooltip
    };
  },
});
</script>

<style scoped>
.map-view {
  --menu-width: 50px;
  height: 500px;
  width: 800px;
  background-color: white;
}

.map-view-content {
  height: calc(100% - var(--header-height));
  display: flex;
}

.el-menu-vertical {
  height: calc(100% - var(--header-height));
  width: var(--menu-width);
  position: absolute;
}

.el-menu-item {
  align-items: center;
  justify-content: center;
}

.map-comp {
  margin-left: var(--menu-width);
  height: 100%;
  width: calc(100% - var(--menu-width));
}
</style>
