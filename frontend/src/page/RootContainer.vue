<template>
  <div class="root-container">
    <div class="main">
      <map-view class="map-view"></map-view>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import MapView from '@/components/views/MapView.vue';

export default defineComponent({
  name: 'RootContainer',
  components: {
    MapView,
  },
  setup() {
    
    return {
      
    };
  },
});
</script>

<style scoped>
.root-container {
  position: absolute;
  height: 900px;
  width: 1850px;
}

.map-view {
  position: relative;
  top: 100px;
}

.main {
  height: 100%;
  width: 100%;
  background-color: rgb(230, 230, 230);
  border-width: 2px;
  border-color: var(--view-border-color);
  border-style: solid;
}
</style>
