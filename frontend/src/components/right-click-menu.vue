<template>
  <div
    :style="{ left: `${left}px`, top: `${top}px` }"
    class="menu"
    v-if="visible"
  >
    <ul>
      <li @click.stop="cancelBrush">取消选区</li>
      <li @click.stop="()=>{}">聚焦已选集合</li>
      <li @click.stop="()=>{}">取消聚焦</li>
    </ul>
  </div>
</template>

<script lang="ts">
import { computed, defineComponent, watch } from "vue";

export default defineComponent({
  name: "RightClickMenue",
  components: {},
  props: {
    visible: Boolean,
    left: Number,
    top: Number,
  },
  emits: ['closeMenu'],
  setup(props, contex) {
    const visible = computed(() => props.visible);

    watch(visible, (value: boolean) => {
      console.log(value)
    })

    const cancelBrush = () => {
      contex.emit('closeMenu')
    }

    return {
      cancelBrush
    };
  },
});
</script>

<style scoped>

.menu {
  margin: 0;
  background: #fff;
  z-index: 3000;
  position: absolute;
  list-style-type: none;
  border-radius: 4px;
  font-size: 19px;
  font-weight: 400;
  color: #333;
  box-shadow: 2px 2px 3px 0 rgba(0, 0, 0, 0.3);
}

.menu ul {
  margin: 0;
  padding: 0;
  list-style: none;
}

.menu li {
  margin: 0;
  padding: 8px 9px 8px 9px;
  cursor: pointer;
}

.menu li:hover {
  background: rgb(226, 238, 255);
  color: rgb(78, 105, 255);
  user-select: none;
}
</style>
