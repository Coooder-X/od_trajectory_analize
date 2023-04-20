import { Ref, ref } from 'vue';

export function useRightClick() {
    const visible: Ref<Boolean> = ref(false);
    const left: Ref<number> = ref(0);
    const top: Ref<number> = ref(0);
    
    function setMenuVisible(value: Boolean) {
        visible.value = value;
    }

    function setLeft(value: number) {
        left.value = value;
    }

    function setTop(value: number) {
        top.value = value;
    }

    return {
        visible, left, top,
        setMenuVisible, setLeft, setTop
    }
}