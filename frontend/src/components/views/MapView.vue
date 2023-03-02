<template>
  <div class="map-view">
    <view-header viewId="A" title="地图视图"></view-header>
    <div class="map-view-content">
      <el-menu
        default-active="2"
        class="el-menu-vertical"
        :collapse="isCollapse"
        @open="handleOpen"
        @close="handleClose"
      >
        <el-menu-item index="2">
          <el-icon><icon-menu /></el-icon>
        </el-menu-item>
        <el-menu-item index="4">
          <el-icon><setting /></el-icon>
        </el-menu-item>
      </el-menu>
      <map-comp class="map-comp"></map-comp>
    </div>
  </div>
</template>

<script>
/* eslint-disable */
import { defineComponent } from "vue";
import ViewHeader from '../ViewHeader'
import MapComp from '../MapComp'
import {
  Document,
  Menu as IconMenu,
  Location,
  Setting,
} from '@element-plus/icons-vue'

export default defineComponent({
  components: { 
    MapComp, 
    ViewHeader,
    Document,
    IconMenu,
    Location,
    Setting
  },
  name: 'MapView',
  props: {
    
  }
})
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
